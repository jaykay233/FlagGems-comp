# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MetaX fast path for the single-token (M == 1) case of ``linear``.

The problem
-----------
``flag_gems.ops.linear.linear`` launches ``linear_kernel`` with

    grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))

At M == 1 -- every decode step -- the first grid axis is always 1, so the
program count collapses to ``cdiv(N, BLOCK_N)``. For the MiniCPM5-2B qkv
projection (N=2560) with BLOCK_N=128 that is 20 programs on a 104-SM MetaX
C500: roughly 20% occupancy, and the kernel runs at ~125 GB/s against a read
ceiling well above 1 TB/s. The kernel also calls ``tl.dot`` on a
BLOCK_M-sized tile of which exactly one row is live, masking off 15/16 of the
MMA work.

``torch.profiler`` on the real model shows the four per-layer projections
summing to 17.1 ms per decode step at M=1, so this is worth attacking.

The fix
-------
A GEMV kernel specialised for M == 1:

* parallelises over N only, so a small BLOCK_N yields N/BLOCK_N programs and
  fills the device;
* drops ``tl.dot`` entirely -- at M == 1 there is no MMA-shaped work, so it
  accumulates ``w[BLOCK_N, BLOCK_K] * a[BLOCK_K]`` in registers and reduces
  once at the end;
* keeps each program's byte volume contiguous, so loads stay coalesced;
* uses ``libentry`` so the launch path skips Triton's per-call specialisation,
  which matters because a decode step issues ~170 of these calls.

Measured on MiniCPM5-2B shapes, bf16, M=1, versus the generic kernel:

    shape            generic      gemv     speedup   achieved bandwidth
    qkv_proj        0.0839 ms  0.0414 ms    2.03x   125 -> 254 GB/s
    o_proj          0.0843 ms  0.0388 ms    2.17x   100 -> 216 GB/s
    gate_up_proj    0.1090 ms  0.0670 ms    1.63x   462 -> 752 GB/s
    down_proj       0.1306 ms  0.0496 ms    2.63x   193 -> 507 GB/s
    lm_head         0.5001 ms  0.4033 ms    1.24x  1069 -> 1326 GB/s

Accuracy is unchanged. The Frobenius relative error against an fp32 reference
is 1.6e-3 for both kernels: the accumulation is the same sum of products in
fp32, only the parallel decomposition differs.

Anything that is not a single-row linear falls straight through to the generic
implementation, so prefill and batched decode are untouched.

This module overrides ``linear`` for the MetaX backend only, through the
mechanism in ``_metax/ops/__init__.py``, where ``SpecOpRegistrar`` rebinds
``flag_gems.linear`` before ``_FULL_CONFIG`` is built.
"""

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops.linear import linear as _generic_linear
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

__all__ = ["linear"]

# Tuned on MetaX C500 (104 SMs). The small BLOCK_N is what buys the
# parallelism; BLOCK_K=512 amortises the loop while keeping the fp32
# accumulator at a reasonable 8 x 512 per program.
_BLOCK_N = 8
_BLOCK_K = 512
_NUM_WARPS = 4

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Kill switch for A/B benchmarking. Resolved once at import so the hot path
# only pays a global lookup.
_ENABLE_GEMV = os.environ.get("FLAG_GEMS_METAX_GEMV", "1") != "0"

# Route M > 1 through the MetaX ``mm`` instead of the generic ``linear_kernel``.
#
# Device time on the MiniCPM5-2B projection shapes (169 GEMMs per step, bf16):
#     M=64   linear_kernel  9.71 ms   vs  mm_kernel_nt  7.24 ms   (1.34x)
#     M=2048 linear_kernel 99.58 ms   vs  mm_kernel_nt 71.01 ms   (1.40x)
# The generic kernel has no MetaX-specific tuned config (tune_configs.yaml has no
# ``linear:`` section), so it cannot adapt its tile to M the way mcblas does.
#
# But the win is device-side only, and ``flag_gems.mm`` costs ~110 us/call on the
# host against ~50 us for ``linear_kernel``. Measured eager, per 169-GEMM step:
#     M=64    route ON 18.31 ms  vs OFF 10.56 ms   -> route LOSES (host-bound)
#     M=2048  route ON 71.01 ms  vs OFF 99.58 ms   -> route WINS  (device-bound)
# so the route must be gated, not applied unconditionally.
_ENABLE_MM_ROUTE = os.environ.get("FLAG_GEMS_METAX_LINEAR_MM", "1") != "0"

# Device work (M*N*K) below which the eager route cannot amortise its host cost.
#
# This used to default to 8e9, on the pre-2026-09-26  measurement that
# "M=2048 wins, M<=1024 loses". That measurement was taken with an over-eager
# config pruner which was dropping the large tiles that win in the 128..1023
# range, so the route looked unprofitable there when in fact the *kernels* were
# mis-selected. After relaxing that pruner (see _prune_mm_dense_configs in mm.py)
# an eager A/B over 51 (shape, M) points - M from 8 to 2048, including the
# smallest realistic shapes - gives `mm` faster in **51/51**, by 1.06x to 1.96x.
# The smallest point measured is M=8, N=768, K=768 (work 4.7e6), where mm is
# still 1.31x faster, so there is no evidence for a floor at all.
#
# Default is therefore 0: route every batched bias-free 2-D projection through
# mm. Set FLAG_GEMS_METAX_MM_MIN_WORK to reintroduce a floor if a workload ever
# shows otherwise.
_MM_ROUTE_MIN_WORK = int(os.environ.get("FLAG_GEMS_METAX_MM_MIN_WORK", "0"))

try:
    _is_capturing = torch.cuda.is_current_stream_capturing
except AttributeError:  # pragma: no cover - older torch without the query
    _is_capturing = None


def _mm_route_worthwhile(M, N, K):
    """Should this (M, N, K) take the ``mm`` route?

    Under CUDA graph capture the host cost is paid once and replayed for free, so
    the mm kernels' device advantage applies at any M - and that is exactly the
    decode case, where M is a small batch. Outside capture (eager prefill) the
    route only pays off once the device work dwarfs the host overhead.
    """
    if _is_capturing is not None:
        try:
            if _is_capturing():
                return True
        except Exception:  # noqa: BLE001 - never let the probe break a GEMM
            pass
    return M * N * K >= _MM_ROUTE_MIN_WORK

# Reachability probe: when set, records every distinct (M, N, K) that reaches
# this override, so it is visible which callers actually route through
# FlagGems' `linear` at all. Inert otherwise.
_TRACE_PATH = os.environ.get("FLAG_GEMS_METAX_GEMV_TRACE")
_seen_shapes: set[tuple[int, int, int]] = set()


def _trace(dim: int, M: int, N: int, K: int, via: str) -> None:
    if not _TRACE_PATH:
        return
    key = (M, N, K)
    if key in _seen_shapes:
        return
    _seen_shapes.add(key)
    try:
        with open(_TRACE_PATH, "a") as fh:
            fh.write(f"{via} dim={dim} M={M} N={N} K={K}\n")
    except OSError:
        pass


@libentry()
@triton.jit
def _gemv_kernel(
    a_ptr,  # (K,)   contiguous, stride 1 along K
    w_ptr,  # (N, K) stride_wn along N
    b_ptr,  # (N,) or a dummy pointer when BIAS is False
    out_ptr,  # (N,)
    N,
    K,
    stride_wn,
    BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = kb * BLOCK_K + offs_k
        k_mask = k_offs < K
        a_vec = tl.load(a_ptr + k_offs, mask=k_mask, other=0.0)
        w_blk = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + k_offs[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc += w_blk.to(tl.float32) * a_vec.to(tl.float32)[None, :]

    res = tl.sum(acc, axis=1)
    if BIAS:
        res = res + tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs_n, res.to(out_ptr.dtype.element_ty), mask=n_mask)


def _linear_gemv(input, weight, bias, dim, K, N):
    """Single-row linear via the GEMV kernel."""
    if dim == 1:
        a = input.view(1, K)
    else:
        a = input if input.is_contiguous() else input.contiguous()

    output = torch.empty((1, N), device=input.device, dtype=input.dtype)
    grid = (triton.cdiv(N, _BLOCK_N),)
    with torch_device_fn.device(input.device):
        _gemv_kernel[grid](
            a,
            weight,
            weight if bias is None else bias,
            output,
            N,
            K,
            weight.stride(0),
            BIAS=bias is not None,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
            num_warps=_NUM_WARPS,
        )
    return output.squeeze(0) if dim == 1 else output


def linear(input, weight, bias=None):
    """``y = x @ weight.T + bias``.

    Single-row inputs take a dedicated GEMV kernel; everything else goes
    straight to the generic FlagGems implementation.

    The guard is deliberately first and cheap: batched and prefill calls --
    i.e. every M > 1 call -- must not pay for a fast path they cannot use.
    """
    if (
        _ENABLE_GEMV
        and weight.dim() == 2
        and input.dim() <= 2
        and input.dtype in _SUPPORTED_DTYPES
        and weight.dtype in _SUPPORTED_DTYPES
    ):
        dim = input.dim()
        if dim == 2:
            M, K = input.shape
        elif dim == 1:
            M, K = 1, input.shape[0]
        else:
            M = 0
        if M == 1 and K > 0 and weight.shape[0] > 0:
            N = weight.shape[0]
            if input.stride(-1) == 1 and weight.stride(-1) == 1:
                logger.debug("GEMS_METAX LINEAR (gemv)")
                _trace(dim, M, N, K, "gemv")
                return _linear_gemv(input, weight, bias, dim, K, N)

    logger.debug("GEMS_METAX LINEAR (generic)")
    if weight.dim() == 2 and input.dim() in (1, 2):
        _M = 1 if input.dim() == 1 else input.shape[0]
        _trace(input.dim(), _M, weight.shape[0], weight.shape[1], "generic")

    # Batched (M > 1) bias-free projections: the MetaX ``mm`` kernels are
    # measurably better than the generic ``linear_kernel`` (see _ENABLE_MM_ROUTE).
    # ``mm(x, w.t())`` is exactly ``linear(x, w)`` when bias is None, and w.t()
    # is the transposed view the ``mm_kernel_nt`` variant wants. Bias is left to
    # the generic kernel so we do not introduce an extra elementwise pass.
    if (
        _ENABLE_MM_ROUTE
        and bias is None
        and weight.dim() == 2
        and input.dim() == 2
        and input.shape[0] > 1
        and input.dtype in _SUPPORTED_DTYPES
        and weight.dtype in _SUPPORTED_DTYPES
        and input.stride(-1) == 1
        and weight.stride(-1) == 1
        and _mm_route_worthwhile(input.shape[0], weight.shape[0], weight.shape[1])
    ):
        import flag_gems

        _trace(input.dim(), input.shape[0], weight.shape[0], weight.shape[1], "mm")
        return flag_gems.mm(input, weight.t())

    return _generic_linear(input, weight, bias)
