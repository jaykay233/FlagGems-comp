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
"""Sort-free top-p masking.

Top-p masking does not need the vocabulary sorted.  The usual implementation
sorts to obtain a threshold `t` -- the point at which the cumulative probability
reaches `p` -- masks everything below it, and scatters the mask back.  The same
`t` can be found by bucketing the values and scanning one cumulative sum, which
turns an O(V log V) sort per decode step into two linear kernels.

That matters on MetaX, where the sort used by vLLM's PyTorch fallback
(flag_gems' radix `compute_global_hist_kernel` + 8x `sweep`) costs 21-28 ms per
step at the batch sizes serving actually uses, making it the dominant kernel in
decoding.  Splitting the work into two kernels rather than a chain of small
torch ops also matters: each flag_gems op costs roughly 240 us of dispatch, so
six small ops on a (rows, n_bins) tensor cost more than one kernel that does all
of it.

Timings below are vocab 130560, p=0.95, single call per row, fp32, measured on
MetaX with flag_gems enabled, and include the comparison the numbers are meant
to replace:

    rows    sort (ms)   top_p_threshold (ms)
       1        2.83                  0.14
      48       21.48                  0.32
      64       28.09                  0.38
     256      113.34                  3.05

The rows=256 entry is the fallback path; see `_V2_MAX_ROWS`.  Above that cutoff
the v2 kernels' (rows, n_bins) atomic histogram leaves L2 and the original
kernels win, so the op keeps both and picks per call.

Accuracy, against the sort-based reference on real serving logits: the retained
set has mask IoU 0.996-0.998 with the reference and the retained probability
mass matches to 6 decimals.  Where the two differ it is only in *which* tokens
sit at the boundary, and always in the same direction: bucketing lands the
threshold on the lower edge of the deciding bucket, so this keeps the
conservative set.  Retained mass is therefore >= p and never < p, which is the
property callers should rely on.  The cost is a slightly looser cut than the
reference: the retained count can be up to ~0.2% larger.  That slack is set by
`n_bins` (a narrower bucket narrows the cut) and is independent of `rows`.

    >>> import torch, flag_gems
    >>> flag_gems.enable()
    >>> logits = torch.randn(8, 130560, device="cuda")
    >>> p = torch.full((8,), 0.95, device="cuda")
    >>> masked = flag_gems.top_p_threshold(logits, p)
"""

from __future__ import annotations

import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["top_p_threshold"]

# Buckets are uniform in log2-probability, so the resolution follows the softmax
# weighting rather than being spread evenly across a range that is almost all
# negligible.  8192 buckets over 40 decades is enough to place the cut within
# ~0.2% in retained count on real logits.  `lo` also bounds how far below the
# row maximum a kept token may sit.
_DEFAULT_N_BINS = 8192
_DEFAULT_LO = -40.0
_DEFAULT_HI = 0.0
_LN2 = math.log(2.0)

# The v2 path below (2-D grid, no int64 index buffer, fused mask) beats the
# original one-program-per-row path up to a row count that keeps its
# (rows, n_bins) atomic histogram resident in L2.  Measured crossover on the
# C500, vocab 130560: v2 wins 2.65x at rows=64, 2.30x at 128, 1.19x at 192, and
# loses at 256 (0.89x) where the histogram spills to HBM and the per-element
# atomics dominate the scatter_add_ they replaced.  Above the cutoff the
# original kernels run unchanged, so no row count is slower than it was.
_V2_MAX_ROWS = 128
_V2_BLOCK = 4096

# Scratch is keyed by shape so a decode loop reuses it instead of reallocating
# ~100 MB per step.
_scratch: dict = {}
_scratch_v2: dict = {}


@triton.jit
def _top_p_mass_kernel(
    logits,
    e_out,
    idx_out,
    amax_out,
    n_cols,
    stride_r,
    N_BINS: tl.constexpr,
    LO: tl.constexpr,
    STEP: tl.constexpr,
    INV_LN2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per row: row max, then exp and bucket index.

    Reading the row twice is deliberate -- the max has to be known before exp
    can be computed safely, and a fused online-softmax plus histogram was
    measured to be slower than this pair of bandwidth-bound passes.
    """
    row = tl.program_id(0)
    base = row * stride_r

    row_max = float("-inf")
    for off in tl.range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(logits + base + cols, mask=cols < n_cols, other=float("-inf"))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    # An all -inf row must keep the reference behaviour: every relative value
    # becomes -inf, so its mass is 0 and no token is selected.
    row_max = tl.where(row_max > -3.0e38, row_max, 0.0)
    tl.store(amax_out + row, row_max)

    for off in tl.range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(logits + base + cols, mask=mask, other=float("-inf"))
        e = tl.exp(x - row_max)
        bucket = ((x - row_max) * INV_LN2 - LO) / STEP
        bucket = tl.minimum(tl.maximum(bucket, 0.0), N_BINS - 1.0)
        tl.store(e_out + base + cols, e, mask=mask)
        # int64 because that is what scatter_add_ requires; converting here
        # avoids materialising an intermediate int64 tensor in torch.
        tl.store(idx_out + base + cols, bucket.to(tl.int64), mask=mask)


@triton.jit
def _top_p_threshold_kernel(
    hist,
    amax,
    p_ptr,
    thresh_out,
    N_BINS: tl.constexpr,
    LO: tl.constexpr,
    STEP: tl.constexpr,
    LN2: tl.constexpr,
):
    """One program per row: reversed cumsum, locate the cut, emit the threshold."""
    row = tl.program_id(0)
    bins = tl.arange(0, N_BINS)
    h = tl.load(hist + row * N_BINS + bins).to(tl.float32)
    total = tl.sum(h, axis=0)
    # rev[i] = mass of buckets [i, end).  The deciding bucket always holds a
    # large share of the mass (rev ~= p * total, and p >= 0.5 in practice), so
    # the flip-cumsum-flip has no cancellation to worry about.
    rev = tl.flip(tl.cumsum(tl.flip(h, 0), axis=0), 0)
    ok = rev >= tl.load(p_ptr + row) * total
    # rev is decreasing, so the last set index is the deciding bucket.  ok[0] is
    # always set because the target can never exceed the total, which also means
    # a zero default is safe here.
    index = tl.max(tl.where(ok, bins, 0), axis=0)
    tl.store(thresh_out + row,
             tl.load(amax + row) + (LO + index.to(tl.float32) * STEP) * LN2)


def _get_scratch(logits: torch.Tensor, n_bins: int) -> dict:
    key = (logits.shape[0], logits.shape[1], logits.device, logits.dtype, n_bins)
    buf = _scratch.get(key)
    if buf is None:
        rows, n_cols = logits.shape
        buf = {
            "e": torch.empty(rows, n_cols, device=logits.device, dtype=torch.float32),
            "idx": torch.empty(rows, n_cols, device=logits.device, dtype=torch.int64),
            "hist": torch.empty(rows, n_bins, device=logits.device, dtype=torch.float32),
            "amax": torch.empty(rows, device=logits.device, dtype=torch.float32),
            "thresh": torch.empty(rows, device=logits.device, dtype=torch.float32),
            "p": torch.empty(rows, device=logits.device, dtype=torch.float32),
        }
        _scratch[key] = buf
    return buf


# ---------------------------------------------------------------------------
# v2 path.
#
# Same algorithm, but shaped for the row counts serving actually uses (1 to the
# concurrency limit).  Four changes, none semantic:
#
#   1. No int64 index buffer.  The bucket is a pure function of `e`
#      (`bucket = (log2(e) - lo) / step`), so materialising 8 B/element and
#      reading it back was redundant.  The fp32 exp/log2 round-trip moves the
#      bin by <1e-3 bins, which cannot change which bucket a token lands in
#      except for tokens sitting exactly on a boundary.
#   2. 2-D grid (rows, col_blocks).  The original used one program per row for
#      both the max pass and the exp pass, so at rows=1 -- the batch-1 case --
#      a single program walked 130k elements twice: measured 11 GB/s, 234 us.
#   3. The max pass stores per-block partials and the mass pass reduces them
#      (n_blocks is ~32), instead of a second serial pass over the row.
#   4. The mask is fused into one compare+select kernel.  `masked_fill` leaves
#      torch to materialise a bool mask tensor first: 10 B/element at 237 GB/s.
#
# The threshold search is also reformulated to avoid two `tl.flip`s of n_bins
# elements: `sum_{j>=i} h_j >= p*total` is equivalent to
# `exclusive_cumsum[i] <= (1-p)*total`, and the count of satisfied bins is the
# answer.  That also removes a p == 1.0 quirk: the original tested
# `rev[0] >= total`, comparing two fp32 cumulative sums that need not round to
# the same value, so it could drop tokens that were nowhere near the cutoff.
# `exclusive_cumsum[0]` is exactly 0 and always satisfies `<= 0`, so the cutoff
# lands on the lowest bucket.  Tokens more than `lo` decades below the row max
# are still excluded at p == 1.0 -- that bound is inherent to bucketing and
# applies to both paths; measured 24 of 8.3M tokens at rows=64.
# ---------------------------------------------------------------------------


@triton.jit
def _top_p_rowmax_v2_kernel(logits, pmax, n_cols, stride_r, NBLK, BLOCK: tl.constexpr):
    """Per-block partial row maxima, grid (rows, NBLK)."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(logits + row * stride_r + cols, mask=m, other=float("-inf"))
    tl.store(pmax + row * NBLK + blk, tl.max(x, axis=0))


@triton.jit
def _top_p_mass_v2_kernel(
    logits, pmax, hist, n_cols, stride_r, NBLK,
    NBLK_P2: tl.constexpr, N_BINS: tl.constexpr, LO: tl.constexpr,
    STEP: tl.constexpr, INV_LN2: tl.constexpr, BLOCK: tl.constexpr,
):
    """Resolve the row max from the partials, then accumulate exp into buckets.

    Grid (rows, NBLK).  Each element costs one read and one atomic add into the
    (rows, n_bins) histogram; nothing else is written.
    """
    row = tl.program_id(0)
    blk = tl.program_id(1)
    bi = tl.arange(0, NBLK_P2)
    partials = tl.load(pmax + row * NBLK + bi, mask=bi < NBLK, other=float("-inf"))
    row_max = tl.max(partials, axis=0)
    # An all -inf row must keep the reference behaviour: every relative value
    # becomes -inf, so its mass is 0 and no token is selected.
    row_max = tl.where(row_max > -3.0e38, row_max, 0.0)

    cols = blk * BLOCK + tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(logits + row * stride_r + cols, mask=m, other=float("-inf"))
    e = tl.exp(x - row_max)
    bucket = ((x - row_max) * INV_LN2 - LO) / STEP
    bucket = tl.minimum(tl.maximum(bucket, 0.0), N_BINS - 1.0)
    tl.atomic_add(hist + row * N_BINS + bucket.to(tl.int32), e, mask=m)


@triton.jit
def _top_p_amax_v2_kernel(pmax, amax, NBLK, NBLK_P2: tl.constexpr):
    row = tl.program_id(0)
    bi = tl.arange(0, NBLK_P2)
    p = tl.load(pmax + row * NBLK + bi, mask=bi < NBLK, other=float("-inf"))
    mx = tl.max(p, axis=0)
    tl.store(amax + row, tl.where(mx > -3.0e38, mx, 0.0))


@triton.jit
def _top_p_threshold_v2_kernel(
    hist, amax, p_ptr, thresh_out,
    N_BINS: tl.constexpr, LO: tl.constexpr, STEP: tl.constexpr, LN2: tl.constexpr,
):
    """One program per row: forward exclusive cumsum, count, emit the cutoff."""
    row = tl.program_id(0)
    bins = tl.arange(0, N_BINS)
    h = tl.load(hist + row * N_BINS + bins).to(tl.float32)
    total = tl.sum(h, axis=0)
    excl = tl.cumsum(h, axis=0) - h
    target = (1.0 - tl.load(p_ptr + row)) * total
    index = tl.sum((excl <= target).to(tl.int32), axis=0) - 1
    tl.store(thresh_out + row,
             tl.load(amax + row) + (LO + index.to(tl.float32) * STEP) * LN2)


@triton.jit
def _top_p_mask_v2_kernel(src, dst, thresh, n_cols, stride_r, BLOCK: tl.constexpr):
    """dst = src where src >= thresh[row] else -inf, grid (rows, NBLK).

    Row-major rather than a flat element grid: n_cols is not a multiple of
    BLOCK, so a flat block would straddle two rows and read the wrong threshold.
    """
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(src + row * stride_r + cols, mask=m, other=0.0)
    t = tl.load(thresh + row)
    tl.store(dst + row * stride_r + cols, tl.where(x < t, float("-inf"), x), mask=m)


def _get_scratch_v2(logits: torch.Tensor, n_bins: int, nblk: int) -> dict:
    key = (logits.shape[0], logits.shape[1], logits.device, logits.dtype, n_bins)
    buf = _scratch_v2.get(key)
    if buf is None:
        rows, _ = logits.shape
        buf = {
            "pmax": torch.empty(rows, nblk, device=logits.device, dtype=torch.float32),
            "hist": torch.empty(rows, n_bins, device=logits.device, dtype=torch.float32),
            "amax": torch.empty(rows, device=logits.device, dtype=torch.float32),
            "thresh": torch.empty(rows, device=logits.device, dtype=torch.float32),
            "p": torch.empty(rows, device=logits.device, dtype=torch.float32),
        }
        _scratch_v2[key] = buf
    return buf


def _top_p_threshold_v2(
    logits: torch.Tensor, p: torch.Tensor, n_bins: int, lo: float, step: float,
) -> torch.Tensor:
    rows, n_cols = logits.shape
    block = _V2_BLOCK
    nblk = triton.cdiv(n_cols, block)
    nblk_p2 = triton.next_power_of_2(nblk)
    buf = _get_scratch_v2(logits, n_bins, nblk)
    buf["p"].copy_(p.to(device=logits.device, dtype=torch.float32))

    out = torch.empty_like(logits)
    grid = (rows, nblk)
    _top_p_rowmax_v2_kernel[grid](
        logits, buf["pmax"], n_cols, logits.stride(0), nblk, block)
    buf["hist"].zero_()
    _top_p_mass_v2_kernel[grid](
        logits, buf["pmax"], buf["hist"], n_cols, logits.stride(0), nblk,
        nblk_p2, n_bins, lo, step, 1.0 / _LN2, block)
    _top_p_amax_v2_kernel[(rows,)](
        buf["pmax"], buf["amax"], nblk, nblk_p2)
    _top_p_threshold_v2_kernel[(rows,)](
        buf["hist"], buf["amax"], buf["p"], buf["thresh"], n_bins, lo, step, _LN2)
    _top_p_mask_v2_kernel[grid](
        logits, out, buf["thresh"], n_cols, logits.stride(0), block)
    return out


def top_p_threshold(
    logits: torch.Tensor,
    p: torch.Tensor,
    n_bins: int = _DEFAULT_N_BINS,
    lo: float = _DEFAULT_LO,
    hi: float = _DEFAULT_HI,
    block: int = 4096,
) -> torch.Tensor:
    """Mask `logits` for top-p sampling without sorting.

    Equivalent to vLLM's `apply_top_k_top_p_pytorch(logits, None, p)`: keeps the
    smallest set of highest-probability tokens whose probability mass reaches
    `p`, and returns a new tensor with the rest set to -inf.  `logits` is not
    modified.

    Args:
        logits: 2-D float32 tensor of shape (rows, vocab).
        p: scalar tensor or 1-D tensor of length `rows`, in (0, 1].
        n_bins: histogram resolution.  Higher places the cut more tightly, at
            the cost of the small per-row histogram; it does not affect the
            bandwidth-bound passes.
        lo, hi: bucket range in log2-probability, relative to the row maximum.

    Returns:
        A tensor of the same shape and dtype, with tokens below the threshold
        set to -inf.

    Raises:
        ValueError: if the input is not a 2-D float32 CUDA tensor, `p` has an
            incompatible shape, or the last dimension is not contiguous.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D, got shape {tuple(logits.shape)}")
    if logits.dtype != torch.float32:
        raise ValueError(f"logits must be float32, got {logits.dtype}")
    if not logits.is_cuda:
        raise ValueError("logits must be on a CUDA device")
    if not logits.is_contiguous():
        # A non-contiguous last dimension would break the row addressing.
        if logits.stride(-1) != 1:
            raise ValueError("logits must have a contiguous last dimension")
        logits = logits.contiguous()

    rows, n_cols = logits.shape
    if n_cols == 0:
        return logits.clone()

    p = torch.as_tensor(p)
    if p.numel() == 1:
        p = p.reshape(1).expand(rows)
    elif p.numel() != rows:
        raise ValueError(
            f"p must be scalar or have {rows} elements, got {p.numel()}")

    n_bins = int(n_bins)
    step = (hi - lo) / n_bins
    # Serving rows are bounded by the concurrency limit; below the measured
    # crossover the v2 path is faster by 2.6x (rows=64) to 3.3x (rows=1).
    if rows <= _V2_MAX_ROWS:
        return _top_p_threshold_v2(logits, p, n_bins, lo, step)
    buf = _get_scratch(logits, n_bins)
    buf["p"].copy_(p.to(device=logits.device, dtype=torch.float32))

    _top_p_mass_kernel[(rows,)](
        logits, buf["e"], buf["idx"], buf["amax"], n_cols, logits.stride(0),
        n_bins, lo, step, 1.0 / _LN2, block,
    )
    # sum over the histogram equals sum(exp) exactly, every token lands in a
    # bucket, so the total mass comes out of the histogram for free.
    buf["hist"].zero_()
    buf["hist"].scatter_add_(1, buf["idx"], buf["e"])
    _top_p_threshold_kernel[(rows,)](
        buf["hist"], buf["amax"], buf["p"], buf["thresh"],
        n_bins, lo, step, _LN2,
    )
    return logits.masked_fill(logits < buf["thresh"].unsqueeze(1), float("-inf"))
