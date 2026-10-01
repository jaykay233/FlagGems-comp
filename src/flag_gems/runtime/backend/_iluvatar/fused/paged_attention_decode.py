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

"""Decode paged attention via ixformer flashinfer (HND KV layout).

vLLM's TRITON_ATTN KV cache is BNDH ``[num_blocks, page_size, kv_heads, head_dim]``.
ixformer ``paged_attention_flashinfer`` expects HND
``[num_blocks, kv_heads, page_size, head_dim]``.  This wrapper converts layout,
builds flashinfer page metadata from ``block_table`` / ``seqused_k``, and writes
into the caller-provided ``out`` buffer.
"""

from __future__ import annotations

import torch


def _bndh_to_hnd(cache: torch.Tensor) -> torch.Tensor:
    # [B, N, H, D] -> [B, H, N, D]
    # Ixformer needs HND/BHND logical indexing and contiguous head_dim,
    # but accepts strided pages. Keep vLLM BNDH storage to avoid copying the
    # full KV cache on each prefill/decode attention call.
    return cache.permute(0, 2, 1, 3)


def _build_flashinfer_pages(
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (indptr, indices, last_page_len) for flashinfer paged attention.

    Stays free of host synchronization: this runs once per attention layer, so
    a single data-dependent shape (which forces a device-to-host sync) costs
    more than the kernel saves.  ``indices`` is therefore built by scattering
    into a fixed upper-bound buffer, with out-of-range pages routed to a sink
    slot.  flashinfer only reads ``indices[indptr[i]:indptr[i + 1]]``, so the
    slack beyond ``indptr[-1]`` is never observed.
    """
    device = seqused_k.device
    num_seqs, max_pages = block_table.shape
    pages = (seqused_k + (page_size - 1)) // page_size
    indptr = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    torch.cumsum(pages.to(torch.int32), dim=0, out=indptr[1:])

    page_ids = torch.arange(max_pages, device=device, dtype=pages.dtype)
    valid = page_ids.unsqueeze(0) < pages.unsqueeze(1)
    dest = indptr[:-1].unsqueeze(1).to(torch.int64) + page_ids.unsqueeze(0)
    sink = num_seqs * max_pages
    dest = torch.where(valid, dest, torch.full_like(dest, sink))
    scattered = torch.zeros(sink + 1, dtype=torch.int32, device=device)
    scattered.scatter_(
        0, dest.reshape(-1), block_table.to(torch.int32).reshape(-1)
    )
    indices = scattered[:sink]

    last_page_len = seqused_k % page_size
    last_page_len = torch.where(
        last_page_len == 0,
        torch.full_like(last_page_len, page_size),
        last_page_len,
    ).to(torch.int32)
    return indptr, indices, last_page_len


def paged_attention_decode(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    max_seqlen_k: int = -1,
) -> torch.Tensor:
    """Run ixformer decode attention for a pure decode batch.

    Args:
        out: ``[num_tokens, num_heads, head_dim]`` (num_tokens == batch for decode).
        query: same shape as ``out``.
        key_cache / value_cache: BNDH paged KV.
        block_table: ``[batch, max_blocks]``.
        seqused_k: ``[batch]`` KV lengths.
        softmax_scale: attention scale.
        max_seqlen_k: optional hint; derived from ``seqused_k`` when ``<= 0``.
    """
    from ixformer.inference.functions.paged_attention import (
        paged_attention_flashinfer,
    )

    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError("paged_attention_decode expects 4D BNDH KV caches")
    page_size = key_cache.shape[1]
    if max_seqlen_k is None or max_seqlen_k <= 0:
        max_seqlen_k = int(seqused_k.max().item())

    k_hnd = _bndh_to_hnd(key_cache)
    v_hnd = _bndh_to_hnd(value_cache)
    indptr, indices, last_page_len = _build_flashinfer_pages(
        block_table, seqused_k, page_size
    )
    paged_attention_flashinfer(
        out,
        query,
        (k_hnd, v_hnd),
        indptr,
        indices,
        last_page_len,
        float(softmax_scale),
        max_seq_len=max_seqlen_k,
        kv_cache_format="HND",
    )
    return out
