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
"""Fused top-p threshold + exponential-noise sampling.

vLLM samples after top-p with

    ids = (softmax(masked_logits) / q).argmax(-1),   q ~ Exponential(1)

The softmax denominator is a positive per-row constant, so it cannot change
the argmax.  Dropping it turns the whole tail -- softmax, the full-vocabulary
probability tensor, the probability divide and the final argmax -- into one
pass that reads ``logits`` and ``q`` once and writes one index per row.

Public API: ``top_p_sample(logits, p, q) -> int64 [rows]``.  The caller must
generate ``q`` exactly as vLLM's ``random_sample`` does (``empty_like`` +
``exponential_()``, plus any per-request generators).

This reuses the threshold machinery of :mod:`flag_gems.ops.top_p_threshold`
(row max -> histogram in log2-probability -> cutoff), then does the masked
exponential weighting and the argmax in a single blocked pass.  It is the
v2 (rows <= 128) path only; above that the caller should keep the original
softmax + ``random_sample``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.ops.top_p_threshold import (
    _DEFAULT_HI,
    _DEFAULT_LO,
    _DEFAULT_N_BINS,
    _LN2,
    _V2_BLOCK,
    _V2_MAX_ROWS,
    _get_scratch_v2,
    _top_p_amax_v2_kernel,
    _top_p_mass_v2_kernel,
    _top_p_rowmax_v2_kernel,
    _top_p_threshold_v2_kernel,
)

__all__ = ["top_p_sample"]


# Per-row block partials for the sampled index.  Keyed by shape/device so a
# decode loop reuses them instead of allocating every step.
_scratch: dict = {}


def _get_sample_scratch(logits: torch.Tensor, nblk: int) -> dict:
    key = (logits.shape[0], logits.shape[1], logits.device)
    buf = _scratch.get(key)
    if buf is None:
        rows = logits.shape[0]
        buf = {
            "sval": torch.empty(rows, nblk, device=logits.device, dtype=torch.float32),
            "sidx": torch.empty(rows, nblk, device=logits.device, dtype=torch.int32),
        }
        _scratch[key] = buf
    return buf


@triton.jit
def _top_p_sample_partial_kernel(
    logits, q, thresh, amax, sval, sidx, n_cols, stride_r,
    INV_LN2: tl.constexpr, BLOCK: tl.constexpr,
):
    """Per-block max of ``exp(x - rowmax) / q`` over the top-p kept set.

    Kept tokens are ``x >= thresh`` (the same strict cut as the mask kernel).
    The per-row softmax denominator is omitted: it is a positive constant and
    argmax is scale invariant.
    """
    row = tl.program_id(0)
    blk = tl.program_id(1)
    nblk = tl.num_programs(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    m = cols < n_cols
    # logits may be bf16 (the caller skipped vLLM's fp32 cast); bf16 is a
    # subset of fp32, so widening here is exact.
    x = tl.load(logits + row * stride_r + cols, mask=m, other=float("-inf")).to(
        tl.float32)
    noise = tl.load(q + row * stride_r + cols, mask=m, other=1.0)
    t = tl.load(thresh + row)
    mx = tl.load(amax + row)
    e = tl.exp2((x - mx) * INV_LN2)
    score = tl.where(m & (x >= t), e / noise, float("-inf"))
    tl.store(sval + row * nblk + blk, tl.max(score, axis=0))
    tl.store(
        sidx + row * nblk + blk,
        (blk * BLOCK + tl.argmax(score, axis=0)).to(tl.int32),
    )


@triton.jit
def _top_p_sample_reduce_kernel(
    sval, sidx, out, nblk, NBLK_P2: tl.constexpr,
):
    row = tl.program_id(0)
    bi = tl.arange(0, NBLK_P2)
    m = bi < nblk
    v = tl.load(sval + row * nblk + bi, mask=m, other=float("-inf"))
    best = tl.argmax(v, axis=0)
    idx = tl.load(sidx + row * nblk + best)
    tl.store(out + row, idx.to(tl.int64))


def top_p_sample(
    logits: torch.Tensor,
    p,
    q: torch.Tensor,
    n_bins: int = _DEFAULT_N_BINS,
    lo: float = _DEFAULT_LO,
    hi: float = _DEFAULT_HI,
    block: int = _V2_BLOCK,
) -> torch.Tensor:
    """Fused top-p filter + exponential-noise sample.

    Statistically equivalent to vLLM's
    ``(top_p_threshold(logits, p).softmax(-1) / q).argmax(-1)``.

    Args:
        logits: 2-D float32/bfloat16 CUDA tensor ``(rows, vocab)``.
        p: scalar tensor or 1-D tensor of length ``rows`` in (0, 1].
        q: float32 tensor shaped like ``logits``, holding Exponential(1) noise.
        n_bins, lo, hi, block: top-p bucketing parameters (see
            :func:`flag_gems.top_p_threshold`).

    Returns:
        int64 tensor ``(rows,)`` of sampled token ids.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D, got shape {tuple(logits.shape)}")
    if logits.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"logits must be float32 or bfloat16, got {logits.dtype}")
    if not logits.is_cuda:
        raise ValueError("logits must be on a CUDA device")
    if logits.stride(-1) != 1:
        raise ValueError("logits must have a contiguous last dimension")
    if not logits.is_contiguous():
        logits = logits.contiguous()
    # q stays float32 even when logits are bf16: it is Exponential(1) noise and
    # an 8-bit mantissa would change near-tie selection.
    if q.shape != logits.shape or q.dtype != torch.float32:
        raise ValueError("q must be float32 with the same shape as logits")
    if not q.is_contiguous():
        q = q.contiguous()

    rows, n_cols = logits.shape
    if n_cols == 0 or rows == 0:
        return torch.empty(rows, dtype=torch.int64, device=logits.device)
    if rows > _V2_MAX_ROWS:
        raise ValueError(
            f"top_p_sample supports rows <= {_V2_MAX_ROWS}, got {rows}")

    p = torch.as_tensor(p)
    if p.numel() == 1:
        p = p.reshape(1).expand(rows)
    elif p.numel() != rows:
        raise ValueError(
            f"p must be scalar or have {rows} elements, got {p.numel()}")

    n_bins = int(n_bins)
    step = (hi - lo) / n_bins
    nblk = triton.cdiv(n_cols, block)
    nblk_p2 = triton.next_power_of_2(nblk)

    buf = _get_scratch_v2(logits, n_bins, nblk)
    buf["p"].copy_(p.to(device=logits.device, dtype=torch.float32))

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
        buf["hist"], buf["amax"], buf["p"], buf["thresh"],
        n_bins, lo, step, _LN2)

    sbuf = _get_sample_scratch(logits, nblk)
    _top_p_sample_partial_kernel[grid](
        logits, q, buf["thresh"], buf["amax"], sbuf["sval"], sbuf["sidx"],
        n_cols, logits.stride(0), 1.0 / _LN2, block)

    out = torch.empty(rows, dtype=torch.int64, device=logits.device)
    _top_p_sample_reduce_kernel[(rows,)](
        sbuf["sval"], sbuf["sidx"], out, nblk, nblk_p2)
    return out
