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
"""Fused ``gate_up_proj + silu_and_mul`` for the MetaX backend.

Motivation
----------
The MLP gate/up projection is normally two kernels::

    gate_up = x @ W.T          # (M, K) x (2N, K) -> (M, 2N)
    out     = silu(gate_up[:, :N]) * gate_up[:, N:]   # -> (M, N)

Measured on the C500 (2026-09-26) the elementwise second pass costs 21% of the
pair at decode (17.2us on top of a 63.4us GEMM for N=6144, K=2048) and 11% at
prefill, purely on the 2N write followed by the 2N read.

This module folds both into one kernel
--------------------------------------
Each program owns an output tile of ``(BLOCK_M, BLOCK_N)`` **in the N-wide
result space**. For that tile it accumulates *two* dot products against the same
A tile: the gate rows ``W[rn]`` and the up rows ``W[rn + N]``. The activation is
then applied in registers and only the N-wide result is stored, so the 2N
intermediate never exists:

    acc_gate = sum_k A[:, k] * W[rn,      k]
    acc_up   = sum_k A[:, k] * W[rn + N,  k]
    C        = silu(acc_gate) * acc_up

Grid ordering (``GROUP_M`` swizzle), the ``EVEN_M``/``EVEN_N``/``EVEN_K``
masking and the tile shapes deliberately mirror ``mm_kernel_nt`` so the two
kernels can share tuned configs and measurement history.

Cost model: the A tile is reused for two dots (better arithmetic intensity than
``mm``), but two B tiles must be resident per stage, so the shared-memory
footprint is higher and the viable tile set is narrower.
"""

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

EXPAND_CONFIG_FILENAME = "tune_configs.yaml"


# Tiles live in ``_metax/tune_configs.yaml`` under ``mm_silu_mul``. They are
# narrower than ``mm_nt``'s because each program must keep BOTH the gate and the
# up B tile resident (see the module docstring), which doubles the per-stage
# shared-memory cost. The fallback below is only used if the YAML entry is
# missing (e.g. an older install of this package).
_FALLBACK_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "pipeline": "basic"},
        num_stages=3,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "pipeline": "basic"},
        num_stages=2,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "pipeline": "basic"},
        num_stages=3,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "pipeline": "basic"},
        num_stages=2,
        num_warps=4,
    ),
]


def _tuned_configs():
    try:
        configs = runtime.get_tuned_config("mm_silu_mul")
    except Exception:  # noqa: BLE001 - never let config loading break the op
        configs = None
    if not configs:
        return _FALLBACK_CONFIGS
    return configs


_SMEM_LIMIT = 64 * 1024


def _prune_silu_mul_configs(configs, named_args, **kwargs):
    """Drop tiles whose two-B-tile footprint cannot fit the C500's 64KB smem.

    ``LibTuner`` already treats an OutOfResources compile as a non-candidate, so
    this pruner is an optimisation (it avoids paying a failed compile) rather
    than a correctness requirement. The estimate assumes every operand tile is
    live for ``num_stages`` at once, which is the conservative end of what
    Triton's pipeliner does.
    """
    configs = list(configs)
    kept = []
    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        block_n = config.kwargs["BLOCK_N"]
        block_k = config.kwargs["BLOCK_K"]
        stages = config.num_stages
        # bf16: 2 bytes. A tile + gate B tile + up B tile, per stage.
        stage_bytes = (block_m * block_k + 2 * block_k * block_n) * 2 * stages
        if stage_bytes > _SMEM_LIMIT:
            continue
        kept.append(config)
    return kept or configs


@libentry()
@libtuner(
    configs=_tuned_configs(),
    key=["M", "N", "K"],
    strategy=["align32_geometric", "align32", "align32"],
    prune_configs_by={"early_config_prune": _prune_silu_mul_configs},
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_silu_mul",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit
def mm_silu_mul_kernel_nt(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """``C = silu(A @ B[:N].T) * (A @ B[N:2N].T)`` for ``A:(M,K)``, ``B:(2N,K)``."""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if EVEN_M:
        ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    else:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if EVEN_N:
        rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    else:
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + ram[:, None] * K + rk[None, :]
    # B is (2N, K) row-major: gate lives in rows [0, N), up in rows [N, 2N).
    b_gate_ptrs = B + rk[:, None] + rbn[None, :] * K
    b_up_ptrs = B + rk[:, None] + (rbn[None, :] + N) * K

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b_gate = tl.load(b_gate_ptrs)
            b_up = tl.load(b_up_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=rk[None, :] < k_remaining, other=0.0)
            b_gate = tl.load(b_gate_ptrs, mask=rk[:, None] < k_remaining, other=0.0)
            b_up = tl.load(b_up_ptrs, mask=rk[:, None] < k_remaining, other=0.0)
        if a.dtype != b_gate.dtype:
            a = a.to(C.dtype.element_ty)
            b_gate = b_gate.to(C.dtype.element_ty)
            b_up = b_up.to(C.dtype.element_ty)
        acc_gate = tl.dot(a, b_gate, acc_gate, out_dtype=tl.float32, allow_tf32=False)
        acc_up = tl.dot(a, b_up, acc_up, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_gate_ptrs += BLOCK_K
        b_up_ptrs += BLOCK_K

    # silu(gate) * up, computed in fp32 then cast once on store.
    silu_gate = acc_gate / (1.0 + tl.exp(-acc_gate))
    result = (silu_gate * acc_up).to(C.dtype.element_ty)

    c_ptrs = C + rm[:, None] * N + rn[None, :]
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, result)
    else:
        tl.store(c_ptrs, result, mask=(rm < M)[:, None] & (rn < N)[None, :])


def mm_silu_mul(x, w):
    """``silu(x @ w[:N].T) * (x @ w[N:].T)``, fused.

    Args:
        x: ``(M, K)`` input, row-major (contiguity is enforced).
        w: ``(2N, K)`` gate/up weight, row-major, gate rows first.

    Returns:
        ``(M, N)`` tensor of the input dtype.
    """
    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()
    if w.stride(0) > 1 and w.stride(1) > 1:
        w = w.contiguous()
    assert x.dim() == 2 and w.dim() == 2, "mm_silu_mul expects 2-D inputs"
    M, K = x.shape
    two_n, wk = w.shape
    assert wk == K, "incompatible dimensions"
    assert two_n % 2 == 0, "gate/up weight must have an even row count"
    N = two_n // 2

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(x.device):
        mm_silu_mul_kernel_nt[grid](x, w, out, M, N, K, GROUP_M=8)
    return out


# --- Dispatch -----------------------------------------------------------------
#
# Fusing is not a strict win. It removes the 2N write/read round-trip, which is
# pure memory traffic, so it pays off exactly where the GEMM is bandwidth-bound:
# decode, with a small M. Once M grows the GEMM becomes compute-bound and the
# large tiles that win are the ones the fused kernel *cannot* use, because it
# has to keep two B tiles resident and the C500 caps shared memory at 64KB.
#
# Measured on the C500 (2026-09-26, K=2048, W=(12288,2048) i.e. N_half=6144),
# fused vs FlagGems' own two-pass (`flag_gems.linear` + `silu_and_mul`):
#   M=16   -> fused 62.9us vs 129.0us   fused 2.05x faster
#   M=64   -> fused 73.1us vs 130.5us   fused 1.79x faster
#   M=128  -> fused 103.5us vs 133.4us  fused 1.29x faster
#   M=256  -> fused 124.5us vs 146.8us  fused 1.18x faster
#   M=512  -> fused 225.0us vs 219.6us  two-pass 1.02x faster
#   M=2048 -> fused 870us  vs 785us    two-pass 1.11x faster
#
# The M=1 case is special: `linear` takes the dedicated GEMV path there and is
# faster than the fused kernel, so the dispatcher's `M <= MAX_M` guard is paired
# with the GEMV fast path living in linear.py.
_MM_SILU_MUL_MAX_M = int(os.environ.get("FLAG_GEMS_METAX_MM_SILU_MUL_MAX_M", "256"))
_FORCE = os.environ.get("FLAG_GEMS_METAX_MM_SILU_MUL", "").strip()


def _two_pass(x, w):
    """Reference formulation: a plain ``mm`` followed by ``silu_and_mul``."""
    import flag_gems

    two_n = w.shape[0]
    n = two_n // 2
    gate_up = flag_gems.mm(x, w.t())  # (M, 2N)
    return flag_gems.silu_and_mul(gate_up[:, :n], gate_up[:, n:])


def silu_and_mul_gate_up(x, w):
    """``silu(x @ w[:N].T) * (x @ w[N:].T)``, picking the better kernel for ``M``.

    Args:
        x: ``(M, K)`` activation, row-major.
        w: ``(2N, K)`` gate/up weight, row-major, gate rows first.

    Returns:
        ``(M, N)`` tensor of the input dtype.

    Set ``FLAG_GEMS_METAX_MM_SILU_MUL=fused|two_pass`` to pin a path, or
    ``FLAG_GEMS_METAX_MM_SILU_MUL_MAX_M`` to move the crossover.
    """
    assert x.dim() == 2 and w.dim() == 2, "expected 2-D input and weight"
    assert w.shape[1] == x.shape[1], "incompatible dimensions"
    assert w.shape[0] % 2 == 0, "gate/up weight must have an even row count"
    M = x.shape[0]
    if _FORCE == "fused":
        return mm_silu_mul(x, w)
    if _FORCE == "two_pass":
        return _two_pass(x, w)
    if M <= _MM_SILU_MUL_MAX_M:
        return mm_silu_mul(x, w)
    return _two_pass(x, w)
