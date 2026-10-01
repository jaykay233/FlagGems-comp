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

"""Prefill / mixed paged attention via ixformer flash_attn_varlen (BHND).

vLLM BNDH ``[num_blocks, page_size, kv_heads, head_dim]`` is permuted to the
BHND layout ``[num_blocks, kv_heads, page_size, head_dim]`` that
``ixformer.contrib.vllm_flash_attn.flash_attn_varlen_func`` accepts with
``block_table=...``.  Contiguous (non-paged) layouts are not handled here.
"""

from __future__ import annotations

import torch


def _bndh_to_bhnd(cache: torch.Tensor) -> torch.Tensor:
    # Same permute as HND for this 4D layout naming: [B, N, H, D] -> [B, H, N, D]
    # Ixformer needs HND/BHND logical indexing and contiguous head_dim,
    # but accepts strided pages. Keep vLLM BNDH storage to avoid copying the
    # full KV cache on each prefill/decode attention call.
    return cache.permute(0, 2, 1, 3)


def _cu_seqlens_from_seqused(seqused: torch.Tensor) -> torch.Tensor:
    # zeros() rather than empty() + ``cu[0] = 0``: assigning a Python scalar
    # into a CUDA tensor synchronizes, once per layer per step.
    cu = torch.zeros(
        seqused.shape[0] + 1, dtype=torch.int32, device=seqused.device
    )
    torch.cumsum(seqused.to(torch.int32), dim=0, out=cu[1:])
    return cu


def flash_attn_prefill(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
) -> torch.Tensor:
    """Run ixformer paged flash-attn for prefill / mixed batches.

    Args:
        out / query: ``[total_q_tokens, num_heads, head_dim]``.
        key_cache / value_cache: BNDH paged KV.
        cu_seqlens_q: ``[batch + 1]`` query prefix sums.
        seqused_k: ``[batch]`` KV lengths (used to build ``cu_seqlens_k``).
        block_table: ``[batch, max_blocks]``.
    """
    from ixformer.contrib.vllm_flash_attn.flash_attn_interface import (
        flash_attn_varlen_func,
    )

    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError("flash_attn_prefill expects 4D BNDH KV caches")
    if block_table is None:
        raise ValueError("flash_attn_prefill requires a paged block_table")

    k_bhnd = _bndh_to_bhnd(key_cache)
    v_bhnd = _bndh_to_bhnd(value_cache)
    cu_seqlens_k = _cu_seqlens_from_seqused(seqused_k)

    flash_attn_varlen_func(
        query,
        k_bhnd,
        v_bhnd,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        window_size=window_size,
        softcap=float(softcap or 0.0),
        block_table=block_table,
        out=out,
    )
    return out
