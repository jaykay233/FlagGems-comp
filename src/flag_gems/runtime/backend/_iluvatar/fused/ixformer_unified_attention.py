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
from .paged_attention_decode import paged_attention_decode, paged_attention_mtp
from .paged_decode_splitk import paged_decode_splitk
from .paged_decode_triton import paged_decode_triton

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
    "decode_splitk": 0,
    "split_splitk": 0,
    "decode_fa_pad": 0,
    "decode_ivcore": 0,
    "split_fa_pad": 0,
    "decode_triton": 0,
    "spec_mtp": 0,
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
    "decode_splitk": False,
    "split_splitk": False,
    "decode_fa_pad": False,
    "split_fa_pad": False,
    "split": False,
    "split_unsafe_skip": False,
    "spec_mtp": False,
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
# flashinfer is correct at batch>=32.  For 10<=n<32, padding q/bt/sk to 32
# recovers the fast correct kernel and beats split-K on long ctx (n=30/16k:
# ~0.81 ms vs ~1.68 ms).  Below PAD_MIN, split-K is still faster.  Eager only:
# CUDA-graph capture sizes stay on split-K to avoid dynamic pad buffers in
# the graph.  Override: VLLM_FL_ILUVATAR_DECODE_FA_PAD=0 / _PAD_MIN.
# With sk=0 pad rows (flashinfer skips empty slots) PAD beats splitk from
# n≈6 @4k/16k; was 10 when pad rows repeated the last seq's full KV.
_DECODE_FA_PAD_MIN = 6
_FA_PAD_WS: dict = {}
# Bumped in set_batch_split so a reused block_table buffer with mutated
# contents cannot keep stale padded pages across steps.  Within a step the
# cache keys on data_ptr+shape (not id()) because the split path passes a
# fresh ``block_table[:n_dec]`` view every layer.
_FA_PAD_GEN = 0


def _decode_is_unsafe(batch: int, max_seqlen_k: int) -> bool:
    """Both arguments are host-side already, so this costs no sync."""
    return batch < _DECODE_MIN_BATCH and max_seqlen_k >= _DECODE_UNSAFE_CTX


def _fa_pad_enabled() -> bool:
    return os.getenv("VLLM_FL_ILUVATAR_DECODE_FA_PAD", "1") != "0"


def _fa_pad_min() -> int:
    raw = os.getenv("VLLM_FL_ILUVATAR_DECODE_FA_PAD_MIN")
    if raw is None:
        return _DECODE_FA_PAD_MIN
    try:
        return max(1, int(raw))
    except ValueError:
        return _DECODE_FA_PAD_MIN


def _fa_pad_target() -> int:
    """Launch batch for the padded FA path.

    Default equals ``_DECODE_MIN_BATCH`` (32): the smallest size at which the
    vendor flashinfer decode kernel is correct.  Override with
    ``VLLM_FL_ILUVATAR_DECODE_FA_PAD_TARGET`` (must be >= 32) to experiment with
    higher occupancy at 16k's live batch of ~29 — pad rows still carry
    ``seqused_k=0`` so they add no KV traffic.  Values below 32 are clamped.
    """
    raw = os.getenv("VLLM_FL_ILUVATAR_DECODE_FA_PAD_TARGET")
    if raw is None:
        return _DECODE_MIN_BATCH
    try:
        return max(_DECODE_MIN_BATCH, int(raw))
    except ValueError:
        return _DECODE_MIN_BATCH


def _should_fa_pad(batch: int, max_seqlen_k: int) -> bool:
    """Pad-to-target FA instead of split-K for mid-size unsafe decode batches."""
    if not _fa_pad_enabled():
        return False
    if not _decode_is_unsafe(batch, max_seqlen_k):
        return False
    if batch < _fa_pad_min():
        return False
    # Avoid allocating / reshaping pad buffers inside a CUDA graph capture.
    if torch.cuda.is_current_stream_capturing():
        return False
    return True


def _paged_attention_decode_padded(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    max_seqlen_k: int,
) -> None:
    """Run flashinfer decode at ``_fa_pad_target()``, write back ``out``[:n].

    Within a step, ``block_table`` / ``seqused_k`` are stable across layers, so
    padded bt/sk and flashinfer page metadata are built once and reused.  Only
    ``query`` (and the result slice) are copied per layer — the previous
    per-layer ``_build_flashinfer_pages`` scatter was what erased the microbench
    win in e2e (§17.37).

    Cache key uses ``data_ptr``+shape (not ``id``): the split path passes a
    fresh ``block_table[:n_dec]`` view every layer, which shares storage with
    the parent.  ``_FA_PAD_GEN`` (bumped in ``set_batch_split``) invalidates
    across steps so in-place KV-table updates cannot reuse stale pages.
    """
    from .paged_attention_decode import _build_flashinfer_pages

    n = int(query.shape[0])
    target = _fa_pad_target()
    assert 0 < n < target
    device = query.device
    n_heads, head_dim = int(query.shape[1]), int(query.shape[2])
    max_pages = int(block_table.shape[1])
    page_size = int(key_cache.shape[1])
    key = (device, target, n_heads, head_dim, query.dtype, block_table.dtype)
    ws = _FA_PAD_WS.get(key)
    if ws is None or ws["bt"].shape[1] < max_pages:
        ws = {
            "q": torch.zeros(
                target, n_heads, head_dim, device=device, dtype=query.dtype
            ),
            "out": torch.empty(
                target, n_heads, head_dim, device=device, dtype=query.dtype
            ),
            "bt": torch.zeros(
                target, max_pages, device=device, dtype=block_table.dtype
            ),
            "sk": torch.zeros(target, device=device, dtype=seqused_k.dtype),
            "gen": -1,
            "meta_key": None,
            "pages": None,
        }
        _FA_PAD_WS[key] = ws
    q_p, out_p, bt_p, sk_p = ws["q"], ws["out"], ws["bt"], ws["sk"]
    if bt_p.shape[1] != max_pages:
        bt_p = torch.zeros(
            target, max_pages, device=device, dtype=block_table.dtype
        )
        ws["bt"] = bt_p
        ws["gen"] = -1
        ws["pages"] = None

    # Prefix slices share data_ptr with the parent contiguous table.
    meta_key = (
        block_table.data_ptr(),
        tuple(block_table.shape),
        seqused_k.data_ptr(),
        tuple(seqused_k.shape),
        n,
    )
    if (
        ws["gen"] != _FA_PAD_GEN
        or ws["meta_key"] != meta_key
        or ws["pages"] is None
    ):
        # Real rows first; pad rows keep sk=0 / bt=0 so flashinfer skips their
        # KV walk.  Repeating the last seq (old behaviour) forced full-length
        # scans on the pad slots: 0.63 ms -> 0.43 ms at n=10 @16k, which is what
        # moved the crossover down to n>=6 (FINDINGS-iluvatar §17.37.5).
        bt_p.zero_()
        sk_p.zero_()
        bt_p[:n].copy_(block_table)
        sk_p[:n].copy_(seqused_k)
        q_p[n:].zero_()
        ws["pages"] = _build_flashinfer_pages(bt_p, sk_p, page_size)
        ws["gen"] = _FA_PAD_GEN
        ws["meta_key"] = meta_key

    q_p[:n].copy_(query)
    paged_attention_decode(
        out=out_p,
        query=q_p,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=bt_p,
        seqused_k=sk_p,
        softmax_scale=softmax_scale,
        max_seqlen_k=max_seqlen_k,
        pages=ws["pages"],
    )
    out.copy_(out_p[:n])

def _splitk_enabled() -> bool:
    return os.getenv("VLLM_FL_ILUVATAR_SPLITK_DECODE", "1") != "0"


# The Triton flash-decoding backend is vendor-free and, measured dispatch to
# dispatch (same Python entry, only the backend body differs), wins for
# mid-size batches at short-to-medium context and loses on long context,
# where pad-to-32 FA is already running at 641 GB/s:
#
#   ctx 4600:  n=32 1.24x  n=64 1.07x  n=128 1.00x
#   ctx 17408: n=32 0.97x  n=64 0.92x
#
# n<32 is excluded because the vendor path there is pad-to-32 FA, which is
# faster than both (4k/n=29: 0.244 ms vs our 0.281 ms).  The upper bound is
# 144 rather than 64 on purpose: under FULL_DECODE_ONLY the batch a kernel sees
# is the *padded* capture size, so a 64-request step runs with n_batch=72 and a
# 64 cap made most of the scored 4k steps fall back to the vendor path.
#
# Default ON (§17.58–17.59 + 4k e2e A/B 2026-11-06): BNDH row-major matches
# warm FA (~488µs); e2e Total tok/s Triton ≈ FA (6226 vs 6122, no regression).
# =0 falls back to vendor FA; =auto uses the shape gate below.
def _triton_decode_mode() -> str:
    return os.getenv("VLLM_FL_ILUVATAR_DECODE_TRITON", "1").strip().lower()


def _triton_decode_enabled() -> bool:
    return _triton_decode_mode() not in ("0", "off", "false", "no")


def _triton_decode_wanted(n_batch: int, max_seqlen_k: int) -> bool:
    mode = _triton_decode_mode()
    if mode in ("0", "off", "false", "no"):
        return False
    if mode in ("1", "on", "true", "yes"):
        return True
    return 32 <= n_batch <= 144 and max_seqlen_k <= 8192


def _splitk_is_usable(q: torch.Tensor, k: torch.Tensor, out: torch.Tensor) -> bool:
    """Shapes the split-K decode kernel indexes directly.

    It walks the KV cache with explicit strides but assumes the head dimension
    is contiguous and a power of two, and that the GQA group divides evenly.
    """
    head_dim = q.shape[-1]
    if head_dim & (head_dim - 1):
        return False
    if q.stride(-1) != 1 or out.stride(-1) != 1 or k.stride(-1) != 1:
        return False
    num_kv_heads = k.shape[2]
    return num_kv_heads > 0 and q.shape[1] % num_kv_heads == 0


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
    # Host-side min(q_len) from query_start_loc_cpu (no GPU D2H). -1 = unset.
    "min_q_len": -1,
    "batch": -1,
}


def set_batch_split(
    num_decodes: int,
    num_decode_tokens: int,
    num_q_tokens: int,
    decode_pages: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    prefill_cu_k: Optional[torch.Tensor] = None,
    prefill_cu_q: Optional[torch.Tensor] = None,
    min_q_len: int = -1,
    batch: int = -1,
) -> None:
    """Publish the decode/prefill boundary of the current step's batch.

    ``num_q_tokens`` is carried so the consumer can verify the stash belongs to
    the batch it was handed, rather than to some other KV cache group.
    Optional ``decode_pages`` / ``prefill_cu_*`` are the once-per-step metadata
    kernels; omit them and every layer rebuilds (the old behaviour).
    ``min_q_len`` is from host ``query_start_loc_cpu`` so the FA shape gate
    never needs ``cu_seqlens_q.min().item()`` (GPU D2H) on the hot path.
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
    _BATCH_SPLIT["min_q_len"] = int(min_q_len)
    _BATCH_SPLIT["batch"] = int(batch)
    _GRAPH_PAGES["key"] = None
    _GRAPH_PAGES["pages"] = None
    global _FA_PAD_GEN
    _FA_PAD_GEN += 1


def clear_batch_split() -> None:
    global _FA_PAD_GEN
    _FA_PAD_GEN += 1
    _GRAPH_PAGES["key"] = None
    _GRAPH_PAGES["pages"] = None
    _BATCH_SPLIT["num_decodes"] = -1
    _BATCH_SPLIT["num_decode_tokens"] = -1
    _BATCH_SPLIT["num_q_tokens"] = -1
    _BATCH_SPLIT["decode_pages"] = None
    _BATCH_SPLIT["decode_n"] = -1
    _BATCH_SPLIT["prefill_cu_k"] = None
    _BATCH_SPLIT["prefill_cu_q"] = None
    _BATCH_SPLIT["min_q_len"] = -1
    _BATCH_SPLIT["batch"] = -1


def _min_q_len_from_cpu_cu(query_start_loc_cpu: torch.Tensor) -> int:
    """``min(q_len)`` from a host cu_seqlens; no device sync."""
    if query_start_loc_cpu is None or query_start_loc_cpu.numel() < 2:
        return -1
    # Ensure CPU: a CUDA tensor here would reintroduce the D2H we are removing.
    if query_start_loc_cpu.is_cuda:
        return -1
    q_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    return int(q_lens.min().item())


def prebuild_split_meta(
    num_decodes: int,
    num_decode_tokens: int,
    num_q_tokens: int,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_size: int,
    query_start_loc_cpu: Optional[torch.Tensor] = None,
) -> None:
    """Build flashinfer/FA metadata once and publish it with the split.

    Called from the attention-metadata builder (once per step).  Covers both
    the mixed-batch split and a pure-decode batch (``num_decodes == batch``),
    so eager decode layers also reuse the pages.  When
    ``query_start_loc_cpu`` is supplied, also stash ``min_q_len`` for the
    sync-free FA shape gate.
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
    elif n_dec == 0 and batch > 0:
        # Pure-prefill step: stash cu_seqlens_k once so FA does not rebuild
        # zeros+cumsum on every layer (device work, no D2H, but 42× launches).
        prefill_cu_k = _cu_seqlens_from_seqused(seqused_k)
        prefill_cu_q = cu_seqlens_q
    min_q_len = _min_q_len_from_cpu_cu(query_start_loc_cpu)
    set_batch_split(
        n_dec,
        n_dec_tok,
        int(num_q_tokens),
        decode_pages=decode_pages,
        prefill_cu_k=prefill_cu_k,
        prefill_cu_q=prefill_cu_q,
        min_q_len=min_q_len,
        batch=batch,
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

# Pure-prefill FA shape gate.  Calibrated on jagged_prefill_micro.txt:
# FA wins on few medium/long chunks; loses when n_seq is large and q is short
# (cost tracks sequence count).  ``PREFILL_MIN_Q`` is the *cheap* sync-free
# reject floor for large batches (decode+chunk mixed), not the only admit rule.
# Override floor with VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q.
# Disable the (n_seq, min_q, kv_len) table with
# VLLM_FL_ILUVATAR_IXFORMER_PREFILL_SHAPE_GATE=0 (then only min_q floor).
_DEFAULT_PREFILL_MIN_Q = 256
# Below this, microbench never saw a clear FA win.
_FA_Q_FLOOR = 64
# Only small batches may sync to consider q in [_FA_Q_FLOOR, MIN_Q).
_FA_SYNC_MAX_SEQS = 16


def _spec_mtp_enabled() -> bool:
    return os.getenv("VLLM_FL_ILUVATAR_SPEC_MTP", "1") != "0"


def _spec_q_max() -> int:
    raw = os.getenv("VLLM_FL_ILUVATAR_SPEC_Q_MAX", "4")
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


def _short_spec_batch(max_seqlen_q: int, num_q_tokens: int, batch: int) -> bool:
    """True for a short speculative-verify shaped batch.

    n-gram rarely fills every sequence with a draft, so the batch is often
    mixed q=1 and q=2. ``vllm_paged_attention_mtp`` already takes
    ``cu_seqlens_q``, so uniformity is not required. Cap ``max_seqlen_q`` so
    long / chunked prefills stay on the FA or Triton paths.
    """
    q = int(max_seqlen_q)
    if batch <= 0 or q <= 1 or q > _spec_q_max():
        return False
    # At least one sequence must actually be verifying (q > 1).
    return int(num_q_tokens) > batch


def _prefill_min_q() -> int:
    raw = os.getenv("VLLM_FL_ILUVATAR_IXFORMER_PREFILL_MIN_Q")
    if raw is None:
        return _DEFAULT_PREFILL_MIN_Q
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_PREFILL_MIN_Q


def _prefill_shape_gate_enabled() -> bool:
    return os.getenv("VLLM_FL_ILUVATAR_IXFORMER_PREFILL_SHAPE_GATE", "1") != "0"


def _fa_prefill_shape_ok(n_seq: int, min_q: int, kv_len: int) -> bool:
    """Whether pure-prefill FA should beat Triton for this shape.

    From jagged_prefill_micro (FA/Triton): 8×256 ≈2.5×; 8×64@kv≥1k ≈1.26×;
    8×128 ≈win; 16×256 win; 16×128 ≈1.23× (skip); 32×64 / many short jagged
    ≈0.56× (skip).
    """
    if n_seq <= 0 or min_q < _FA_Q_FLOOR:
        return False
    if n_seq <= 8 and min_q >= 256:
        return True
    if n_seq <= 8 and min_q >= 128:
        return True
    if n_seq <= 8 and min_q >= _FA_Q_FLOOR and int(kv_len) >= 1024:
        return True
    if n_seq <= 16 and min_q >= 256:
        return True
    return False


def _q_lens_from_cu(cu_seqlens_q: torch.Tensor) -> torch.Tensor:
    return cu_seqlens_q[1:] - cu_seqlens_q[:-1]


def _stashed_min_q_len(num_q_tokens: int, batch: int) -> Optional[int]:
    """Host min(q_len) published by ``prebuild_split_meta``, if it matches."""
    if int(_BATCH_SPLIT["num_q_tokens"]) != int(num_q_tokens):
        return None
    stashed_batch = int(_BATCH_SPLIT["batch"])
    if stashed_batch >= 0 and stashed_batch != int(batch):
        return None
    m = int(_BATCH_SPLIT["min_q_len"])
    return m if m >= 0 else None


def _classify_prefill_batch(
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    num_q_tokens: int,
    min_q: int,
    max_seqlen_k: int = 0,
) -> str:
    """Return ``prefill``, ``mixed``, or ``short`` for max_seqlen_q > 1.

    Cheap shape-only tests reject common decode+prefill batches without a
    device->host sync.  Prefer once-per-step host ``min_q_len`` from
    ``query_start_loc_cpu``; only fall back to ``cu_seqlens_q.min().item()``
    when the stash is missing (e.g. split patch off).
    """
    batch = int(cu_seqlens_q.numel()) - 1
    if batch <= 0:
        return "short"

    max_q = int(max_seqlen_q)
    tokens = int(num_q_tokens)
    kv = int(max_seqlen_k)
    use_shape = _prefill_shape_gate_enabled()

    if max_q < _FA_Q_FLOOR:
        return "short"

    # Sync-free: large batch that cannot be all-long under the cheap floor is
    # almost always decode+chunk mixed on the official serve path.
    cheap_floor = min_q if min_q > 0 else _DEFAULT_PREFILL_MIN_Q
    if cheap_floor > 0 and tokens < batch * cheap_floor:
        if batch > _FA_SYNC_MAX_SEQS or not use_shape:
            # max_q below the floor ⇒ many short pure chunks; else mixed.
            return "short" if max_q < cheap_floor else "mixed"
        # Small batch: may be short-jagged pure prefill; prefer stashed min.
        if max_q < cheap_floor and batch > 8:
            return "short"

    # Legacy / shape-gate-off: require max_q >= cheap_floor before reading lens.
    if not use_shape:
        if cheap_floor > 0 and max_q < cheap_floor:
            return "short"
        if cheap_floor > 0 and tokens < batch * cheap_floor:
            return "mixed"

    min_len = _stashed_min_q_len(tokens, batch)
    if min_len is None:
        q_lens = _q_lens_from_cu(cu_seqlens_q)
        min_len = int(q_lens.min().item())
    if min_len == 1:
        return "mixed"
    if use_shape:
        if _fa_prefill_shape_ok(batch, min_len, kv):
            return "prefill"
        return "short"
    if cheap_floor > 0 and min_len < cheap_floor:
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
        n_batch = int(block_table.shape[0])
        # Opt-in ivcore u8 decode (hi-K + INT8-V). Env VLLM_FL_ILUVATAR_IVCORE_DECODE=1.
        try:
            from vllm_fl.dispatch.backends.vendor.iluvatar.ivcore_u8_decode import (
                enabled as _ivcore_on,
                paged_decode_ivcore,
            )
            if _ivcore_on() and paged_decode_ivcore(
                out=out,
                query=q,
                key_cache=k,
                value_cache=v,
                block_table=block_table,
                seqused_k=seqused_k,
                softmax_scale=softmax_scale,
            ):
                _STATS["decode_ivcore"] += 1
                _log_dispatch("decode_ivcore", q, k, reason="ivcore_u8_int8v")
                return True
        except Exception:
            pass
        if _triton_decode_wanted(n_batch, int(max_seqlen_k)):
            # Drop-in replacement for the whole decode branch below, inside the
            # shapes the Triton kernel measurably wins.  It returns False (no
            # side effects) when the shape or the workspace is unusable, so the
            # vendor path remains the fallback.
            if paged_decode_triton(
                out=out,
                query=q,
                key_cache=k,
                value_cache=v,
                block_table=block_table,
                seqused_k=seqused_k,
                softmax_scale=softmax_scale,
            ):
                _STATS["decode_triton"] += 1
                _log_dispatch("decode_triton", q, k, reason="triton_flashdecoding")
                return True
        if _decode_is_unsafe(n_batch, int(max_seqlen_k)):
            # flashinfer is wrong for n<32 (collapses onto seq0 KV).  Prefer
            # pad-to-32 FA when it beats split-K (n>=PAD_MIN); else split-K.
            if _should_fa_pad(n_batch, int(max_seqlen_k)):
                _STATS["decode_fa_pad"] += 1
                _log_dispatch(
                    "decode_fa_pad", q, k, reason=f"pad_{n_batch}_to_{_DECODE_MIN_BATCH}"
                )
                _paged_attention_decode_padded(
                    out=out,
                    query=q,
                    key_cache=k,
                    value_cache=v,
                    block_table=block_table,
                    seqused_k=seqused_k,
                    softmax_scale=softmax_scale,
                    max_seqlen_k=int(max_seqlen_k),
                )
                return True
            if _splitk_enabled() and _splitk_is_usable(q, k, out):
                _STATS["decode_splitk"] += 1
                _log_dispatch("decode_splitk", q, k, reason="small_batch_long_ctx")
                paged_decode_splitk(
                    out=out,
                    query=q,
                    key_cache=k,
                    value_cache=v,
                    block_table=block_table,
                    seqused_k=seqused_k,
                    softmax_scale=softmax_scale,
                )
                return True
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

    # Speculative verify: short q=2..4. FA varlen is ~10x slower here, and
    # Triton would re-read KV per query. MTP scans KV once. Drafts are often
    # sparse across the batch, so cu_seqlens may be mixed q=1/q=2. vLLM's
    # query buffer is frequently a non-contiguous view; ixformer requires
    # contiguous Q/O, so materialize when needed.
    batch = int(block_table.shape[0])
    num_q_tokens = int(q.shape[0])
    if (
        causal
        and _spec_mtp_enabled()
        and _short_spec_batch(max_seqlen_q, num_q_tokens, batch)
    ):
        n_kv = int(k.shape[2])
        n_heads = int(q.shape[1])
        if n_kv > 0 and (n_heads // n_kv) * int(max_seqlen_q) > 4:
            q_m = q if q.is_contiguous() else q.contiguous()
            if out.is_contiguous():
                out_m = out
            else:
                out_m = torch.empty_like(q_m)
            _STATS["spec_mtp"] += 1
            _log_dispatch("spec_mtp", q_m, k, reason=f"short_q={int(max_seqlen_q)}")
            paged_attention_mtp(
                out=out_m,
                query=q_m,
                key_cache=k,
                value_cache=v,
                block_table=block_table,
                seqused_k=seqused_k,
                cu_seqlens_q=cu_seqlens_q,
                softmax_scale=softmax_scale,
                max_seqlen_q=int(max_seqlen_q),
                max_seqlen_k=int(max_seqlen_k),
            )
            if out_m is not out:
                out.copy_(out_m)
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
        # Mixed batches run eager, so unlike pure decode they get no shelter
        # from graph capture.  The decode slice is also smaller than the whole
        # batch, so it can fall into the flashinfer corner even when the batch
        # does not -- in which case the decode rows go to split-K instead.
        decode_unsafe = _decode_is_unsafe(n_dec, int(max_seqlen_k))
        use_fa_pad = decode_unsafe and _should_fa_pad(n_dec, int(max_seqlen_k))
        use_splitk = (
            decode_unsafe
            and not use_fa_pad
            and _splitk_enabled()
            and _splitk_is_usable(q, k, out)
        )
        if decode_unsafe and not use_fa_pad and not use_splitk:
            _STATS["split_unsafe_skip"] += 1
            _log_dispatch(
                "split_unsafe_skip", q, k, reason="decode_slice_below_min_batch"
            )
            return False
        if use_fa_pad:
            stat = "split_fa_pad"
        elif use_splitk:
            stat = "split_splitk"
        else:
            stat = "split"
        _STATS[stat] += 1
        _log_dispatch(stat, q, k, reason=f"decode={n_dec} prefill={batch - n_dec}")
        cu_q_pref = _BATCH_SPLIT["prefill_cu_q"]
        if cu_q_pref is None:
            cu_q_pref = cu_seqlens_q[n_dec:] - cu_seqlens_q[n_dec]
        if use_fa_pad:
            _paged_attention_decode_padded(
                out=out[:n_dec_tok],
                query=q[:n_dec_tok],
                key_cache=k,
                value_cache=v,
                block_table=block_table[:n_dec],
                seqused_k=seqused_k[:n_dec],
                softmax_scale=softmax_scale,
                max_seqlen_k=int(max_seqlen_k),
            )
        elif use_splitk:
            paged_decode_splitk(
                out=out[:n_dec_tok],
                query=q[:n_dec_tok],
                key_cache=k,
                value_cache=v,
                block_table=block_table[:n_dec],
                seqused_k=seqused_k[:n_dec],
                softmax_scale=softmax_scale,
            )
        else:
            paged_attention_decode(
                out=out[:n_dec_tok],
                query=q[:n_dec_tok],
                key_cache=k,
                value_cache=v,
                block_table=block_table[:n_dec],
                seqused_k=seqused_k[:n_dec],
                softmax_scale=softmax_scale,
                max_seqlen_k=max_seqlen_k,
                pages=_cached_decode_pages(n_dec),
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
    # tile64; pure-prefill FA uses (n_seq, min_q, kv_len) shape gate.
    kind = _classify_prefill_batch(
        cu_seqlens_q,
        max_seqlen_q,
        num_q_tokens,
        _prefill_min_q(),
        max_seqlen_k=int(max_seqlen_k),
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
    _log_dispatch("prefill", q, k, reason="shape_gated_pure_prefill_fa")
    # Prefer once-per-step cu_seqlens_k when the stash matches this batch.
    cu_k = None
    if (
        int(_BATCH_SPLIT["num_q_tokens"]) == num_q_tokens
        and int(_BATCH_SPLIT["batch"]) == batch
        and int(_BATCH_SPLIT["num_decodes"]) == 0
    ):
        cu_k = _BATCH_SPLIT["prefill_cu_k"]
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
        cu_seqlens_k=cu_k,
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
