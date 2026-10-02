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

"""vLLM ``unified_attention``-shaped entry that dispatches to ixformer fused ops.

Returns ``True`` when the call was handled, ``False`` when the caller should
fall back to Triton ``unified_attention`` (unsupported feature / layout).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from .flash_attn_prefill import flash_attn_prefill
from .paged_attention_decode import paged_attention_decode

logger = logging.getLogger(__name__)

# Dispatch counters, readable via ``get_dispatch_stats()``.  "layout" counts
# calls rejected because the KV cache is not physically HND, which is the one
# rejection reason that a deployment can fix (VLLM_KV_CACHE_LAYOUT=HND).
# "mixed_skip" / "short_skip" are prefill-shaped calls left on Triton.
_STATS = {
    "decode": 0,
    "prefill": 0,
    "split": 0,
    "mixed_skip": 0,
    "short_skip": 0,
    "decode_unsafe_skip": 0,
    "split_unsafe_skip": 0,
    "unsupported": 0,
    "layout": 0,
}
_HIT_LOG_EVERY = 64
_logged_first = {
    "hit": False,
    "layout": False,
    "unsupported": False,
    "mixed_skip": False,
    "short_skip": False,
    "decode_unsafe_skip": False,
    "split": False,
    "split_unsafe_skip": False,
}


def _split_enabled() -> bool:
    return os.getenv("VLLM_FL_ILUVATAR_IXFORMER_SPLIT", "1") != "0"

# Below 32 sequences with a KV length of at least 2048, ixformer's
# paged_attention_flashinfer reads sequence 0's KV for every row and collapses
# the GQA head mapping; see FINDINGS §17.9.  It is not a precision issue, and
# with peaked attention the error reaches O(1).
#
# Evaluating the KV length here is what keeps the guard cheap.  This function
# runs when the kernel's launch configuration is chosen -- during CUDA graph
# capture, or during an eager call; replays never reach Python.  vLLM captures
# with seq_lens filled to 1, so capture takes the short-context (correct) path
# and bakes it in, and replaying that graph at a real context length stays
# correct (verified in flashinfer_cudagraph_check.txt).  Gating on batch size
# alone would therefore push low-concurrency decode onto Triton for nothing.
# Should vLLM ever capture with real sequence lengths, this still holds: the
# capture call would see the long KV and be refused.
_DECODE_MIN_BATCH = 32
_DECODE_UNSAFE_CTX = 2048


def _decode_is_unsafe(batch: int, max_seqlen_k: int) -> bool:
    """Both arguments are host-side already, so this costs no sync."""
    return batch < _DECODE_MIN_BATCH and max_seqlen_k >= _DECODE_UNSAFE_CTX


# Where the decode rows end in a reordered batch.  ``unified_attention`` only
# receives tensors, so the split has to come from the attention metadata
# builder, which computes it on the host from ``query_start_loc_cpu``.  The
# plugin calls ``set_batch_split`` / ``prebuild_split_meta`` once per step; the
# 42 layers of that step then read it.  Nothing here syncs.
#
# Optional prebuilt tensors (flashinfer pages, FA cu_seqlens) are also stashed
# here.  Under FlagGems those ~20 pointwise ops cost ~6 ms of host time each
# when rebuilt per layer; building them once in the metadata builder drops the
# mixed-step host cost by ~40× (§17.10).
_BATCH_SPLIT: dict = {
    "num_decodes": -1,
    "num_decode_tokens": -1,
    "num_q_tokens": -1,
    "decode_pages": None,  # (indptr, indices, last_page_len) or None
    "decode_n": -1,  # sequences covered by decode_pages
    "prefill_cu_k": None,
    "prefill_cu_q": None,
}


def set_batch_split(
    num_decodes: int,
    num_decode_tokens: int,
    num_q_tokens: int,
    decode_pages: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    prefill_cu_k: Optional[torch.Tensor] = None,
    prefill_cu_q: Optional[torch.Tensor] = None,
) -> None:
    """Publish the decode/prefill boundary of the current step's batch.

    ``num_q_tokens`` is carried so the consumer can verify the stash belongs to
    the batch it was handed, rather than to some other KV cache group.
    Optional ``decode_pages`` / ``prefill_cu_*`` are the once-per-step metadata
    kernels; omit them and every layer rebuilds (the old behaviour).
    """
    _BATCH_SPLIT["num_decodes"] = int(num_decodes)
    _BATCH_SPLIT["num_decode_tokens"] = int(num_decode_tokens)
    _BATCH_SPLIT["num_q_tokens"] = int(num_q_tokens)
    _BATCH_SPLIT["decode_pages"] = decode_pages
    _BATCH_SPLIT["decode_n"] = (
        int(num_decodes) if decode_pages is not None else -1
    )
    _BATCH_SPLIT["prefill_cu_k"] = prefill_cu_k
    _BATCH_SPLIT["prefill_cu_q"] = prefill_cu_q
    _GRAPH_PAGES["key"] = None
    _GRAPH_PAGES["pages"] = None


def clear_batch_split() -> None:
    _GRAPH_PAGES["key"] = None
    _GRAPH_PAGES["pages"] = None
    _BATCH_SPLIT["num_decodes"] = -1
    _BATCH_SPLIT["num_decode_tokens"] = -1
    _BATCH_SPLIT["num_q_tokens"] = -1
    _BATCH_SPLIT["decode_pages"] = None
    _BATCH_SPLIT["decode_n"] = -1
    _BATCH_SPLIT["prefill_cu_k"] = None
    _BATCH_SPLIT["prefill_cu_q"] = None


def prebuild_split_meta(
    num_decodes: int,
    num_decode_tokens: int,
    num_q_tokens: int,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_size: int,
) -> None:
    """Build flashinfer/FA metadata once and publish it with the split.

    Called from the attention-metadata builder (once per step).  Covers both
    the mixed-batch split and a pure-decode batch (``num_decodes == batch``),
    so eager decode layers also reuse the pages.
    """
    from .flash_attn_prefill import _cu_seqlens_from_seqused
    from .paged_attention_decode import _build_flashinfer_pages

    n_dec = int(num_decodes)
    n_dec_tok = int(num_decode_tokens)
    batch = int(block_table.shape[0])
    decode_pages = None
    prefill_cu_k = None
    prefill_cu_q = None
    # threshold-1 reorder: one token per decode row.
    if n_dec > 0 and n_dec_tok == n_dec and n_dec <= batch:
        decode_pages = _build_flashinfer_pages(
            block_table[:n_dec], seqused_k[:n_dec], page_size
        )
        if n_dec < batch:
            prefill_cu_k = _cu_seqlens_from_seqused(seqused_k[n_dec:])
            # Stay on-device: cu_seqlens_q[n_dec] is a 0-d tensor, not a Python int.
            prefill_cu_q = cu_seqlens_q[n_dec:] - cu_seqlens_q[n_dec]
    set_batch_split(
        n_dec,
        n_dec_tok,
        int(num_q_tokens),
        decode_pages=decode_pages,
        prefill_cu_k=prefill_cu_k,
        prefill_cu_q=prefill_cu_q,
    )


def _cached_decode_pages(
    num_seqs: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # The stash is reallocated every step, so a captured graph would replay
    # against freed memory.  Under capture use ``_graph_decode_pages`` instead.
    if torch.cuda.is_current_stream_capturing():
        return None
    if _BATCH_SPLIT["decode_pages"] is None:
        return None
    if int(_BATCH_SPLIT["decode_n"]) != int(num_seqs):
        return None
    return _BATCH_SPLIT["decode_pages"]


# Pages built inside the graph being captured: the first decode layer's build
# kernels are captured (so each replay recomputes them from the persistent
# block_table / seq_lens), and the remaining layers read the same tensors.
# Reset whenever the builder publishes a new step, i.e. before every capture.
_GRAPH_PAGES: dict = {"key": None, "pages": None}


def _graph_decode_pages(
    block_table: torch.Tensor, seqused_k: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .paged_attention_decode import _build_flashinfer_pages

    key = (
        block_table.data_ptr(),
        tuple(block_table.shape),
        seqused_k.data_ptr(),
        int(seqused_k.shape[0]),
        int(page_size),
    )
    if _GRAPH_PAGES["key"] != key:
        _GRAPH_PAGES["pages"] = _build_flashinfer_pages(
            block_table, seqused_k, page_size
        )
        _GRAPH_PAGES["key"] = key
    return _GRAPH_PAGES["pages"]


def _usable_batch_split(num_q_tokens: int, batch: int) -> Optional[tuple[int, int]]:
    """``(num_decodes, num_decode_tokens)`` when the stash matches this call.

    Returns ``None`` unless the batch really is mixed and reordered: a split is
    only usable when there is at least one decode row and at least one prefill
    row, and when the token count proves the stash describes this batch.
    """
    n_dec = _BATCH_SPLIT["num_decodes"]
    n_dec_tok = _BATCH_SPLIT["num_decode_tokens"]
    if _BATCH_SPLIT["num_q_tokens"] != num_q_tokens:
        return None
    if n_dec <= 0 or n_dec >= batch:
        return None
    # With reorder_batch_threshold == 1 every decode row contributes one token;
    # anything else means the batch was reordered under different rules.
    if n_dec_tok != n_dec:
        return None
    return n_dec, n_dec_tok

# Pure-prefill FA only when every sequence's q_len is at least this.
# FA beats Triton on pure chunks down to q=256 (serve_shape_fa_vs_triton.txt),
# so this is not really a quality threshold any more: it is what lets a mixed
# batch be rejected from host-side shapes alone, without a sync.  Values below
# 2 would force the exact check on every call.
# Override with VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q.
_DEFAULT_PREFILL_MIN_Q = 256


def _prefill_min_q() -> int:
    raw = os.getenv("VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q")
    if raw is None:
        return _DEFAULT_PREFILL_MIN_Q
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_PREFILL_MIN_Q


def _q_lens_from_cu(cu_seqlens_q: torch.Tensor) -> torch.Tensor:
    return cu_seqlens_q[1:] - cu_seqlens_q[:-1]


def _classify_prefill_batch(
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    num_q_tokens: int,
    min_q: int,
) -> str:
    """Return ``prefill``, ``mixed``, or ``short`` for max_seqlen_q > 1.

    Only the last check reads ``cu_seqlens_q`` on the host, which costs a
    device->host sync.  Paying that on every mixed step is what the official
    4k serve feels as higher TTFT, so the cheap shape-only tests reject the
    common decode+prefill batches first; ``mixed`` vs ``short`` there is a
    label for diagnostics, both fall back to Triton either way.
    """
    batch = int(cu_seqlens_q.numel()) - 1
    if batch <= 0:
        return "short"
    if min_q > 0:
        if int(max_seqlen_q) < min_q:
            return "short"
        # All q_len >= min_q implies num_q_tokens >= batch * min_q, so a batch
        # failing this cannot be an all-long pure prefill.
        if num_q_tokens < batch * min_q:
            return "mixed"

    q_lens = _q_lens_from_cu(cu_seqlens_q)
    min_len = int(q_lens.min().item())
    if min_len == 1:
        return "mixed"
    if min_q > 0 and min_len < min_q:
        return "short"
    return "prefill"


def get_dispatch_stats() -> dict:
    return dict(_STATS)


def reset_dispatch_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0
    for k in _logged_first:
        _logged_first[k] = False


def _hnd_view_is_free(cache: torch.Tensor) -> bool:
    """True when the BNDH -> HND permute needs no copy.

    ixformer requires contiguous HND pages.  vLLM stores the cache with a
    canonical logical shape whose physical order follows
    ``VLLM_KV_CACHE_LAYOUT``; under HND the permuted view is already
    contiguous.  Under the default NHD it is not, and materializing it would
    copy the whole per-layer cache on every call.
    """
    return cache.permute(0, 2, 1, 3).is_contiguous()


def _log_dispatch(kind: str, q: torch.Tensor, k: torch.Tensor, reason: str = "") -> None:
    total = sum(_STATS.values())
    first_key = "hit" if kind in ("decode", "prefill") else kind
    if first_key not in _logged_first:
        _logged_first[first_key] = False
    is_first = not _logged_first[first_key]
    periodic = kind in ("decode", "prefill") and _STATS[kind] % _HIT_LOG_EVERY == 0
    if not (is_first or periodic):
        return
    _logged_first[first_key] = True
    hnd = k.permute(0, 2, 1, 3)
    logger.warning(
        "ixformer_dispatch %s#%d (total=%d) q=%s k=%s k_strides=%s "
        "hnd_contiguous=%s hnd_strides=%s%s",
        kind,
        _STATS.get(kind, 0),
        total,
        tuple(q.shape),
        tuple(k.shape),
        tuple(k.stride()),
        hnd.is_contiguous(),
        tuple(hnd.stride()),
        f" reason={reason}" if reason else "",
    )


def _window_is_full(window_size) -> bool:
    if window_size is None:
        return True
    if isinstance(window_size, (tuple, list)) and len(window_size) >= 2:
        return int(window_size[0]) < 0 and int(window_size[1]) < 0
    return False


def _is_supported_ixformer_call(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table,
    softcap,
    q_descale,
    alibi_slopes,
    sinks,
    qq_bias,
    mm_prefix_range,
    output_scale,
    use_td: bool,
    kv_quant_mode,
    k_scale_cache,
    v_scale_cache,
    chunk_lookback: int,
    window_size,
) -> bool:
    if block_table is None:
        return False
    if k is None or v is None or k.dim() != 4 or v.dim() != 4:
        return False
    # Ixformer consumes the permuted HND page views directly. Its kernels
    # require head_dim to be contiguous but accept non-contiguous page strides.
    if k.stride(-1) != 1 or v.stride(-1) != 1:
        return False
    if q.dim() != 3:
        return False
    # MiniCPM5 / Llama path: bf16/fp16, head 64/128, no extras.
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False
    head_size = q.shape[-1]
    if head_size not in (64, 128):
        return False
    if softcap not in (None, 0, 0.0):
        return False
    if not _window_is_full(window_size):
        return False
    if alibi_slopes is not None or sinks is not None or qq_bias is not None:
        return False
    if mm_prefix_range is not None or output_scale is not None:
        return False
    if use_td:
        return False
    # vLLM passes k_descale/v_descale as expanded _k_scale/_v_scale even for an
    # unquantized cache, so they carry no information here; the KV dtype below
    # is what decides. A q_descale is only set for an fp8 query.
    if q_descale is not None:
        return False
    if k.dtype not in (torch.bfloat16, torch.float16):
        return False
    if v.dtype is not k.dtype:
        return False
    if k_scale_cache is not None or v_scale_cache is not None:
        return False
    if chunk_lookback is not None and int(chunk_lookback) >= 0:
        return False
    # KVQuantMode.NONE == 0; accept enum or int.
    if kv_quant_mode is not None and int(kv_quant_mode) != 0:
        return False
    return True


def try_ixformer_unified_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size=(-1, -1),
    block_table=None,
    softcap=0.0,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    mm_prefix_range=None,
    use_alibi_sqrt: bool = False,
    kv_quant_mode=0,
    k_scale_cache=None,
    v_scale_cache=None,
    chunk_lookback: int = -1,
    use_td: bool = False,
) -> bool:
    """Attempt ixformer attention. ``True`` = done; ``False`` = use Triton."""
    del (
        seq_threshold_3D,
        num_par_softmax_segments,
        softmax_segm_output,
        softmax_segm_max,
        softmax_segm_expsum,
        k_descale,
        v_descale,
    )
    if use_alibi_sqrt:
        return False
    if not _is_supported_ixformer_call(
        q=q,
        k=k,
        v=v,
        block_table=block_table,
        softcap=softcap,
        q_descale=q_descale,
        alibi_slopes=alibi_slopes,
        sinks=sinks,
        qq_bias=qq_bias,
        mm_prefix_range=mm_prefix_range,
        output_scale=output_scale,
        use_td=use_td,
        kv_quant_mode=kv_quant_mode,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
        chunk_lookback=chunk_lookback,
        window_size=window_size,
    ):
        _STATS["unsupported"] += 1
        _log_dispatch("unsupported", q, k, reason="feature_gate")
        return False

    if not (_hnd_view_is_free(k) and _hnd_view_is_free(v)):
        _STATS["layout"] += 1
        _log_dispatch("layout", q, k, reason="hnd_view_not_contiguous")
        return False

    # Decode: every sequence has q_len == 1.
    if int(max_seqlen_q) == 1:
        if _decode_is_unsafe(int(block_table.shape[0]), int(max_seqlen_k)):
            _STATS["decode_unsafe_skip"] += 1
            _log_dispatch(
                "decode_unsafe_skip", q, k, reason="flashinfer_small_batch_long_ctx"
            )
            return False
        _STATS["decode"] += 1
        _log_dispatch("decode", q, k)
        if torch.cuda.is_current_stream_capturing():
            pages = _graph_decode_pages(block_table, seqused_k, int(k.shape[1]))
        else:
            pages = _cached_decode_pages(int(block_table.shape[0]))
        paged_attention_decode(
            out=out,
            query=q,
            key_cache=k,
            value_cache=v,
            block_table=block_table,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            max_seqlen_k=max_seqlen_k,
            pages=pages,
        )
        return True

    # Mixed batch: if vLLM reordered it, the decode rows form a contiguous
    # prefix and the two halves can go to the kernel each is good at.  FA costs
    # ~0.21 ms per sequence on a mixed batch because every decode row pays a
    # full q-tile sweep, so leaving them together is what made all-FA lose.
    batch = int(block_table.shape[0])
    num_q_tokens = int(q.shape[0])
    split = _usable_batch_split(num_q_tokens, batch)
    if split is not None and _split_enabled():
        n_dec, n_dec_tok = split
        if _decode_is_unsafe(n_dec, int(max_seqlen_k)):
            # Mixed batches run eager, so unlike pure decode they get no
            # shelter from graph capture.  The decode slice is also smaller
            # than the whole batch, so it can fall into the flashinfer corner
            # even when the batch does not.
            _STATS["split_unsafe_skip"] += 1
            _log_dispatch(
                "split_unsafe_skip", q, k, reason="decode_slice_below_min_batch"
            )
            return False
        _STATS["split"] += 1
        _log_dispatch("split", q, k, reason=f"decode={n_dec} prefill={batch - n_dec}")
        pages = _cached_decode_pages(n_dec)
        cu_q_pref = _BATCH_SPLIT["prefill_cu_q"]
        if cu_q_pref is None:
            cu_q_pref = cu_seqlens_q[n_dec:] - cu_seqlens_q[n_dec]
        paged_attention_decode(
            out=out[:n_dec_tok],
            query=q[:n_dec_tok],
            key_cache=k,
            value_cache=v,
            block_table=block_table[:n_dec],
            seqused_k=seqused_k[:n_dec],
            softmax_scale=softmax_scale,
            max_seqlen_k=max_seqlen_k,
            pages=pages,
        )
        flash_attn_prefill(
            out=out[n_dec_tok:],
            query=q[n_dec_tok:],
            key_cache=k,
            value_cache=v,
            cu_seqlens_q=cu_q_pref,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k[n_dec:],
            max_seqlen_k=max_seqlen_k,
            block_table=block_table[n_dec:],
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            cu_seqlens_k=_BATCH_SPLIT["prefill_cu_k"],
        )
        return True

    # Not reordered (or not mixed): mixed or short chunks stay on Triton
    # tile64; only large pure-prefill batches take ixformer FA (§17.6).
    kind = _classify_prefill_batch(
        cu_seqlens_q, max_seqlen_q, num_q_tokens, _prefill_min_q()
    )
    if kind == "mixed":
        _STATS["mixed_skip"] += 1
        _log_dispatch("mixed_skip", q, k, reason="mixed_batch_fallback_triton")
        return False
    if kind == "short":
        _STATS["short_skip"] += 1
        _log_dispatch("short_skip", q, k, reason="short_chunk_fallback_triton")
        return False

    _STATS["prefill"] += 1
    _log_dispatch("prefill", q, k, reason="large_pure_prefill_fa")
    flash_attn_prefill(
        out=out,
        query=q,
        key_cache=k,
        value_cache=v,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size if window_size is not None else (-1, -1),
        softcap=float(softcap or 0.0),
    )
    return True


def ixformer_unified_attention(*args, **kwargs) -> Optional[torch.Tensor]:
    """Convenience wrapper that raises if the call is unsupported."""
    out = kwargs.get("out", args[3] if len(args) > 3 else None)
    ok = try_ixformer_unified_attention(*args, **kwargs)
    if not ok:
        raise RuntimeError(
            "ixformer_unified_attention: unsupported attention configuration"
        )
    return out
