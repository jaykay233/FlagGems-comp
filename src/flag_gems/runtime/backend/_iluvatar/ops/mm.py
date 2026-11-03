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

import logging
import math
import os

import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

from flag_gems import runtime

# Registers the "align32_geometric" key strategy (import side effect).
from flag_gems.runtime.backend._iluvatar.tuner_strategies import (  # noqa: F401
    align32_geometric,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a
    assert a in _ordered_datatypes
    assert b in _ordered_datatypes
    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a
    raise AssertionError("unreachable")


def _to_tl_type(ty):
    return getattr(tl, str(ty).split(".")[-1])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
    # M is data-dependent (chunked prefill feeds arbitrary token counts), so an
    # identity key never warms up: 134 distinct M values were observed in one
    # cache table.  "align32_geometric" folds M into 13 bounded buckets while
    # keeping 32-granularity across the decode region.  N and K are static model
    # dimensions, where plain "align32" is already correct.
    strategy=["align32_geometric", "align32", "align32"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 15,
    },
    warmup=5,
    rep=10,
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
        "UPGRADE": lambda args: math.ceil(
            (args["M"] * args["N"]) / (args["BLOCK_M"] * args["BLOCK_N"])
        ).bit_length()
        > 31,
        "UPGRADE_A_OFFS": lambda args: math.ceil(args["M"] * args["K"]).bit_length()
        > 31,
        "UPGRADE_B_OFFS": lambda args: math.ceil(args["K"] * args["N"]).bit_length()
        > 31,
        "UPGRADE_C_OFFS": lambda args: math.ceil(args["M"] * args["N"]).bit_length()
        > 31,
    }
)
@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    acc_dtype: tl.constexpr,
    input_precision: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    UPGRADE: tl.constexpr,
    UPGRADE_A_OFFS: tl.constexpr,
    UPGRADE_B_OFFS: tl.constexpr,
    UPGRADE_C_OFFS: tl.constexpr,
):
    # matrix multiplication
    if UPGRADE:
        pid = tle.program_id(0)
        pid_z = tle.program_id(1)
    else:
        pid = tl.program_id(0)
        pid_z = tl.program_id(1)
    # grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # # re-order program ID for better L2 performance
    # width = GROUP_M * grid_n
    # group_id = pid // width
    # group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    # pid_m = group_id * GROUP_M + (pid % group_size)
    # pid_n = (pid % width) // (group_size)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    # do matrix multiplication
    if UPGRADE_A_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        ram = (tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if UPGRADE_B_OFFS:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        rbn = (tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)).to(tl.int64)
    else:
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            a = tl.load(A)
            b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
    else:
        loop_num = tl.cdiv(K, BLOCK_K * SPLIT_K) - 1
        for k in range(0, loop_num):
            a = tl.load(A)
            b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk

        _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
        k_remaining = K - loop_num * (BLOCK_K * SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if fp8_fast_accum:
            acc = tl.dot(
                a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
            )
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)

    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    if UPGRADE_C_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


# Skinny-decode hand-picked tiles (ixkn + force-launch sweep, 2026-10-21).
# LibTuner's mm search space starts at BLOCK_M=32 / num_warps=8 and the
# CUDA perf_model often keeps fat tiles that leave BI-V150 at ~12–19% occ /
# ~10–21% TCU for M≈16–32.  These configs beat the autotuned path by
# ~1.2–1.8× on MiniCPM5-2B's four layer GEMMs at M≤32.
# M>32 falls through: M=64 is already ~80–90% of BW ceiling (mm_sweep) and
# the former LE64 hand-picks regressed 4k e2e (5897 vs splitk-only 6027).
# Default on (LE32 only; M>32 falls through). Set FLAGGEMS_ILUVATAR_MM_SKINNY=0 to disable.
# §17.44: 16k +49 tok/s; 4k neutral. Was off after LE64 regression (§17.35).
# Keyed by (N, K); values are (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, warps, stages).
_SKINNY_MM_LE32 = {
    (2560, 2048): (64, 128, 64, 1, 8, 2),  # qkv
    (2048, 2048): (32, 64, 128, 1, 4, 2),  # o_proj
    (12288, 2048): (32, 256, 64, 1, 8, 1),  # gate_up
    (2048, 6144): (32, 64, 128, 1, 8, 2),  # down
}
# 33 < M <= 64 hand-picks (2026-11-03).
#
# Only two of the four layer GEMMs are overridden.  The 4k decode batch lives in
# this M range and the LibTuner path is weak on the two small shapes, but it is
# already good on the two big ones, and forcing those was what caused an earlier
# regression (below).
#
# Measured inside a CUDA graph (42 replays = one layer, i.e. what production
# pays under FULL_DECODE_ONLY), averaged over M = 33/40/48/56/64:
#
#   shape      tuner      forced     gain      x42 layers
#   qkv       27.9 us    20.8 us    +25%      0.31 ms/step
#   o         18.8 us    14.6 us    +22%      0.18 ms/step
#   gate_up   92.0 us    -          tuner wins (546 GB/s = 93% of peak)
#   down      54.8 us    -          tuner wins
#
# qkv and o together are ~0.48 ms off a 35.8 ms 4k decode step (1.35%).
#
# e2e 4k, same session and server flags, only this env var flipping, 3 runs per
# arm:  off 6010.30 / 6022.77 / 6023.97  (mean 6019.01, sd 7.57)
#       on  6031.99 / 6048.16 / 6055.38  (mean 6045.18, sd 11.98)
# The arms do not overlap (on_min - off_max = +8.02), Welch t = 3.2, so
# +26.16 tok/s (+0.43%) is real -- but it is only ~1/3 of the 1.35% step-time
# win, the rest being absorbed by other fixed cost in the step.  An earlier
# 2-run A/B agrees in sign and size: +22.04 tok/s (+0.37%) pooled 5 vs 5.
#
# Both are BLOCK_M=64 so the weight tile is a single DRAM pass, and both keep
# num_stages=2.  o needs num_warps=8: the 4-warp variant is 3% slower in a graph
# even though it looks fine in eager timing.
#
# HISTORY OF THIS CONSTANT -- do not repeat:
#   An earlier version of this dict also forced gate_up (64,128,64,1,8,2) and
#   down (64,64,128,1,4,2), and regressed 4k e2e to 5895.22 tok/s
#   (6010.10/6028.91 baseline, -2.1%).  gate_up forced that way costs 112.5 us
#   vs the tuner's 92.0 us.  The same magnitude as the 2026-10-21 attempt in
#   §17.35 (5897 vs 6027).
#   The root error was benchmarking against _SKINNY_MM_LE32, which is gated to
#   M<=32 and never runs at M=64; the real baseline is mm_kernel's tuner.
#   Set FLAGGEMS_ILUVATAR_MM_SKINNY_LE64=0 to disable.
_SKINNY_MM_LE64 = {
    (2560, 2048): (64, 128, 64, 1, 8, 2),  # qkv
    (2048, 2048): (64, 64, 128, 1, 8, 2),  # o_proj
}
_SKINNY_MM_JIT = None


def _skinny_mm_enabled() -> bool:
    return os.getenv("FLAGGEMS_ILUVATAR_MM_SKINNY", "1") != "0"


def _skinny_mm_le64_enabled() -> bool:
    # On by default: +26.16 tok/s (+0.43%) at 4k, see the table at
    # _SKINNY_MM_LE64.  FLAGGEMS_ILUVATAR_MM_SKINNY_LE64=0 disables.
    return os.getenv("FLAGGEMS_ILUVATAR_MM_SKINNY_LE64", "1") != "0"


def _pick_skinny_mm_cfg(M: int, N: int, K: int):
    if not _skinny_mm_enabled():
        return None
    if M <= 32:
        return _SKINNY_MM_LE32.get((N, K))
    if M <= 64 and _skinny_mm_le64_enabled():
        return _SKINNY_MM_LE64.get((N, K))
    return None


def _mm_jit():
    """Underlying Triton JIT (unwrap LibEntry / LibTuner / Heuristics)."""
    global _SKINNY_MM_JIT
    if _SKINNY_MM_JIT is None:
        _SKINNY_MM_JIT = mm_kernel.fn.fn.fn
    return _SKINNY_MM_JIT


def _launch_mm_fixed(a, b, c, M, N, K, cfg):
    block_m, block_n, block_k, split_k, num_warps, num_stages = cfg
    ab_dtype_tl = _to_tl_type(get_higher_dtype(a.dtype, b.dtype))
    if split_k > 1:
        c.zero_()
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), split_k)
    with torch_device_fn.device(a.device):
        _mm_jit()[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            acc_dtype=tl.float32,
            input_precision=None,
            fp8_fast_accum=True,
            GROUP_M=8,
            AB_DTYPE=ab_dtype_tl,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SPLIT_K=split_k,
            EVEN_K=(K % (block_k * split_k) == 0),
            UPGRADE=False,
            UPGRADE_A_OFFS=False,
            UPGRADE_B_OFFS=False,
            UPGRADE_C_OFFS=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return c


def _launch_mm(a, b, c, M, N, K):
    """Launch Triton matmul _kernel; c must be pre-allocated."""
    cfg = _pick_skinny_mm_cfg(M, N, K)
    if cfg is not None:
        return _launch_mm_fixed(a, b, c, M, N, K, cfg)

    ab_dtype = get_higher_dtype(a.dtype, b.dtype)
    acc_dtype_tl = tl.float32
    ab_dtype_tl = _to_tl_type(ab_dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )

    with torch_device_fn.device(a.device):
        mm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            acc_dtype=acc_dtype_tl,
            input_precision=None,
            fp8_fast_accum=True,
            GROUP_M=8,
            AB_DTYPE=ab_dtype_tl,
        )
    return c


def mm(a, b):
    logger.debug("GEMS_ILUVATAR MM")
    device = a.device
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    return _launch_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    logger.debug("GEMS_ILUVATAR MM_OUT")
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    return _launch_mm(a, b, out, M, N, K)
