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

"""Split-K paged decode attention for BI-V150.

Decode with few sequences and a long context is the worst case on this card.
ixformer's flashinfer kernel is unusable below 32 sequences (FINDINGS §17.9),
and the Triton fallback runs the whole context in one program per (sequence,
KV head): at 16 SMs, 8 sequences x 2 KV heads leaves the card nearly idle and
costs ~4.4 ms per layer at a 17k context.

This kernel splits the KV range instead, so the work scales with the context
rather than the batch, and reduces the partial softmaxes in a second pass.
A KV head's queries are processed together -- MiniCPM5-2B is 16:2 GQA, so one
KV tile serves eight query rows and is read once instead of eight times.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# One program's share of the query rows.  tl.dot wants at least 16 rows, and
# the GQA group is 8, so the tile is padded and the surplus rows are masked.
_BLOCK_M = 16
# Keys per inner iteration.  Default tile is shape-selected in
# ``paged_decode_splitk``: BN=32/w4 for tiny batches (n<=2), BN=16/w2
# otherwise.  ixkn on BI-V150 showed the BN32/w4 kernel stalled ~41% on
# memory dependency with heavy shared-memory bank conflicts; BN16/w2 cuts
# that path ~15–19% at n=4..64 / ctx=4k..16k (ixkn_splitk_retune, 2026-10-21).
# Wider tiles (64+) still lose on the strided KV gather.
_BLOCK_N = 32
_BLOCK_N_WIDE = 32
_BLOCK_N_NARROW = 16
_NUM_WARPS_WIDE = 4
_NUM_WARPS_NARROW = 2
# Tiny decode graphs stay on the wide tile; everything else uses narrow.
_NARROW_MIN_SEQS = 3
# Programs to aim for across the grid.  Raised 256→512 so BN16/w2 keeps
# enough splits at n=24/48 (otherwise those sizes regress vs BN32/w4).
_TARGET_PROGRAMS = 512
_MAX_SPLITS = 32


@triton.jit
def _splitk_partial_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    block_table_ptr,
    seqused_ptr,
    acc_ptr,  # [num_seqs, num_q_heads, num_splits, HEAD_DIM] fp32
    lse_m_ptr,  # [num_seqs, num_q_heads, num_splits] fp32
    lse_l_ptr,  # [num_seqs, num_q_heads, num_splits] fp32
    softmax_scale,
    stride_qs,
    stride_qh,
    stride_kb,
    stride_kp,
    stride_kh,
    stride_vb,
    stride_vp,
    stride_vh,
    stride_bts,
    stride_accs,
    stride_acch,
    stride_accp,
    stride_lses,
    stride_lseh,
    num_splits,
    max_blocks,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)

    seq_len = tl.load(seqused_ptr + seq).to(tl.int32)

    # Even split of the key range; the last split absorbs the remainder.
    chunk = (seq_len + num_splits - 1) // num_splits
    start = split * chunk
    end = tl.minimum(start + chunk, seq_len)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < GROUP

    lse_base = lse_m_ptr + seq * stride_lses + (kv_head * GROUP + offs_m) * stride_lseh + split
    l_base = lse_l_ptr + seq * stride_lses + (kv_head * GROUP + offs_m) * stride_lseh + split

    if start >= end:
        # Empty split: the reduction skips it via l == 0.
        tl.store(lse_base, tl.full([BLOCK_M], float("-inf"), tl.float32), mask=m_mask)
        tl.store(l_base, tl.zeros([BLOCK_M], tl.float32), mask=m_mask)
        return

    q_ptrs = (
        q_ptr
        + seq * stride_qs
        + (kv_head * GROUP + offs_m)[:, None] * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    for block_start in tl.range(start, end, BLOCK_N):
        keys = block_start + offs_n
        valid = keys < end
        # Paged gather: page id from the block table, offset within the page.
        page_idx = keys // PAGE_SIZE
        page_off = keys % PAGE_SIZE
        pages = tl.load(
            block_table_ptr + seq * stride_bts + page_idx,
            mask=page_idx < max_blocks,
            other=0,
        ).to(tl.int32)

        k_ptrs = (
            k_ptr
            + pages[:, None] * stride_kb
            + page_off[:, None] * stride_kp
            + kv_head * stride_kh
            + offs_d[None, :]
        )
        k = tl.load(k_ptrs, mask=valid[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * softmax_scale
        qk = tl.where(valid[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # m_new is -inf only if every key in the tile is masked, which the
        # loop bounds already exclude; guard anyway so exp stays finite.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        p = tl.where(valid[None, :], p, 0.0)

        v_ptrs = (
            v_ptr
            + pages[:, None] * stride_vb
            + page_off[:, None] * stride_vp
            + kv_head * stride_vh
            + offs_d[None, :]
        )
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    acc_ptrs = (
        acc_ptr
        + seq * stride_accs
        + (kv_head * GROUP + offs_m)[:, None] * stride_acch
        + split * stride_accp
        + offs_d[None, :]
    )
    tl.store(acc_ptrs, acc, mask=m_mask[:, None])
    tl.store(lse_base, m_i, mask=m_mask)
    tl.store(l_base, l_i, mask=m_mask)


@triton.jit
def _splitk_reduce_kernel(
    acc_ptr,
    lse_m_ptr,
    lse_l_ptr,
    out_ptr,
    stride_accs,
    stride_acch,
    stride_accp,
    stride_lses,
    stride_lseh,
    stride_os,
    stride_oh,
    num_splits,
    HEAD_DIM: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)

    offs_p = tl.arange(0, BLOCK_P)
    offs_d = tl.arange(0, HEAD_DIM)
    live = offs_p < num_splits

    m = tl.load(
        lse_m_ptr + seq * stride_lses + head * stride_lseh + offs_p,
        mask=live,
        other=float("-inf"),
    )
    l = tl.load(
        lse_l_ptr + seq * stride_lses + head * stride_lseh + offs_p,
        mask=live,
        other=0.0,
    )
    m_max = tl.max(m, axis=0)
    m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
    w = tl.where(l > 0, tl.exp(m - m_max), 0.0)

    acc = tl.load(
        acc_ptr
        + seq * stride_accs
        + head * stride_acch
        + offs_p[:, None] * stride_accp
        + offs_d[None, :],
        mask=live[:, None],
        other=0.0,
    )
    num = tl.sum(acc * w[:, None], axis=0)
    den = tl.sum(l * w, axis=0)
    out = num / tl.where(den > 0, den, 1.0)
    tl.store(out_ptr + seq * stride_os + head * stride_oh + offs_d, out.to(out_ptr.dtype.element_ty))


# Partial buffers must keep their addresses for the lifetime of any CUDA graph
# that captured them (FINDINGS §17.13, lesson 20), so entries are never evicted.
_WORKSPACE: dict = {}


def _workspace(num_seqs: int, num_heads: int, num_splits: int, head_dim: int, device):
    key = (num_seqs, num_heads, num_splits, head_dim, device)
    buf = _WORKSPACE.get(key)
    if buf is None:
        acc = torch.empty(
            (num_seqs, num_heads, num_splits, head_dim), dtype=torch.float32, device=device
        )
        m = torch.empty((num_seqs, num_heads, num_splits), dtype=torch.float32, device=device)
        l = torch.empty((num_seqs, num_heads, num_splits), dtype=torch.float32, device=device)
        buf = (acc, m, l)
        _WORKSPACE[key] = buf
    return buf


def choose_num_splits(num_seqs: int, num_kv_heads: int) -> int:
    """Pick the split count from the batch shape alone.

    It must not depend on the context length: vLLM captures decode graphs with
    ``seq_lens`` filled to 1, and the launch grid is frozen at capture, while
    the real lengths only show up at replay.  Sequence lengths are read inside
    the kernel, so any split count stays correct -- only the occupancy changes.
    """
    programs = max(1, num_seqs * num_kv_heads)
    splits = (_TARGET_PROGRAMS + programs - 1) // programs
    return max(1, min(_MAX_SPLITS, triton.next_power_of_2(splits)))


def paged_decode_splitk(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Decode attention over a paged KV cache, split along the KV range.

    Args:
        out: ``[num_seqs, num_q_heads, head_dim]``.
        query: same shape as ``out``.
        key_cache / value_cache: ``[num_blocks, page_size, num_kv_heads, head_dim]``
            (vLLM's BNDH view; strides may be those of HND storage).
        block_table: ``[num_seqs, max_blocks]`` int32.
        seqused_k: ``[num_seqs]`` int32 KV lengths.
        softmax_scale: attention scale.
        num_splits: override for the split count; derived from the batch shape
            when omitted.
    """
    num_seqs, num_q_heads, head_dim = query.shape
    page_size, num_kv_heads = key_cache.shape[1], key_cache.shape[2]
    group = num_q_heads // num_kv_heads
    if num_splits is None:
        num_splits = choose_num_splits(num_seqs, num_kv_heads)

    # Shape-selected tile: see module constants.  Must depend only on
    # num_seqs (not seq_len) so CUDA-graph capture stays valid.
    if num_seqs < _NARROW_MIN_SEQS:
        block_n, num_warps = _BLOCK_N_WIDE, _NUM_WARPS_WIDE
    else:
        block_n, num_warps = _BLOCK_N_NARROW, _NUM_WARPS_NARROW

    acc, lse_m, lse_l = _workspace(
        num_seqs, num_q_heads, num_splits, head_dim, query.device
    )

    _splitk_partial_kernel[(num_seqs, num_kv_heads, num_splits)](
        query,
        key_cache,
        value_cache,
        block_table,
        seqused_k,
        acc,
        lse_m,
        lse_l,
        softmax_scale,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_table.stride(0),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        lse_m.stride(0),
        lse_m.stride(1),
        num_splits,
        block_table.shape[1],
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        GROUP=group,
        BLOCK_M=max(_BLOCK_M, triton.next_power_of_2(group)),
        BLOCK_N=block_n,
        num_warps=num_warps,
        # One stage: the software pipeline's extra KV buffer costs more shared
        # memory than the strided gather gains from prefetching.
        num_stages=1,
    )
    _splitk_reduce_kernel[(num_seqs, num_q_heads)](
        acc,
        lse_m,
        lse_l,
        out,
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        lse_m.stride(0),
        lse_m.stride(1),
        out.stride(0),
        out.stride(1),
        num_splits,
        HEAD_DIM=head_dim,
        BLOCK_P=triton.next_power_of_2(num_splits),
        num_warps=4,
    )
    return out
