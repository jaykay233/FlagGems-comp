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
from typing import Optional

import torch

from .paged_attention_decode import paged_attention_decode

logger = logging.getLogger(__name__)

# Dispatch counters, readable via ``get_dispatch_stats()``.  "layout" counts
# calls rejected because the KV cache is not physically HND, which is the one
# rejection reason that a deployment can fix (VLLM_KV_CACHE_LAYOUT=HND).
_STATS = {"decode": 0, "prefill": 0, "unsupported": 0, "layout": 0}
_HIT_LOG_EVERY = 64
_logged_first = {"hit": False, "layout": False, "unsupported": False}


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

    # Pure decode only.  Prefill / mixed (max_seqlen_q > 1) stays on the
    # Triton tile64 path: ixformer paged FA wins on full-seq microbenches but
    # loses badly under vLLM chunked-prefill shapes in the official 4k serve.
    if int(max_seqlen_q) != 1:
        _STATS["prefill"] += 1  # counted as "seen but skipped"
        _log_dispatch("prefill", q, k, reason="decode_only_skip")
        return False

    _STATS["decode"] += 1
    _log_dispatch("decode", q, k)
    paged_attention_decode(
        out=out,
        query=q,
        key_cache=k,
        value_cache=v,
        block_table=block_table,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        max_seqlen_k=max_seqlen_k,
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
