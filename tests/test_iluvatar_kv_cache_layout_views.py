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

import pytest
import torch

from flag_gems.runtime.backend._iluvatar.fused.flash_attn_prefill import (
    _bndh_to_bhnd,
)
from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
    _bndh_to_hnd,
)
from flag_gems.runtime.backend._iluvatar.fused.ixformer_unified_attention import (
    _BATCH_SPLIT,
    _cached_decode_pages,
    _classify_prefill_batch,
    _decode_is_unsafe,
    _is_supported_ixformer_call,
    _usable_batch_split,
    clear_batch_split,
    prebuild_split_meta,
    set_batch_split,
)


@pytest.fixture(autouse=True)
def _reset_batch_split():
    clear_batch_split()
    yield
    clear_batch_split()


def test_usable_batch_split_accepts_a_reordered_mixed_batch():
    # 63 decode rows then one 2048-token chunk, as reorder leaves it.
    set_batch_split(num_decodes=63, num_decode_tokens=63, num_q_tokens=2111)
    assert _usable_batch_split(num_q_tokens=2111, batch=64) == (63, 63)


@pytest.mark.parametrize(
    "num_decodes,num_decode_tokens,stashed_tokens,call_tokens,batch,why",
    [
        # Stash belongs to a different batch (other KV cache group, stale step).
        (63, 63, 2111, 2048, 64, "token count mismatch"),
        # Pure decode: nothing to split, the decode path already handles it.
        (64, 64, 64, 64, 64, "no prefill rows"),
        # Pure prefill: no decode prefix.
        (0, 0, 2048, 2048, 2, "no decode rows"),
        # Never published, or cleared after a builder error.
        (-1, -1, -1, 2111, 64, "no stash"),
        # Decode rows carrying more than one token each: not threshold-1 reorder.
        (8, 16, 2048, 2048, 20, "decode tokens exceed decode rows"),
    ],
)
def test_usable_batch_split_rejects(
    num_decodes, num_decode_tokens, stashed_tokens, call_tokens, batch, why
):
    if num_decodes >= 0:
        set_batch_split(num_decodes, num_decode_tokens, stashed_tokens)
    assert _usable_batch_split(call_tokens, batch) is None, why


@pytest.mark.parametrize(
    "batch,max_seqlen_k,expected",
    [
        # Below 32 sequences with a long context, flashinfer reads sequence 0.
        (1, 4096, True),
        (8, 2048, True),
        (31, 8192, True),
        # Enough sequences: correct at any context length.
        (32, 8192, False),
        (64, 4096, False),
        # Short context: correct at any batch size.
        (8, 1024, False),
        (1, 256, False),
        # CUDA graph capture passes seq_lens filled to 1.  That must be allowed
        # through: capture then bakes the short-context path, and replaying it
        # at a real context length stays correct.  Refusing here would cost
        # low-concurrency decode the flashinfer speedup for no safety gain.
        (8, 1, False),
        (1, 1, False),
        # The eager warmup that precedes capture does carry real lengths.
        (8, 5120, True),
    ],
)
def test_decode_is_unsafe(batch, max_seqlen_k, expected):
    assert _decode_is_unsafe(batch, max_seqlen_k) is expected


@pytest.mark.parametrize(
    "q_lens,min_q,expected",
    [
        ([512, 512], 512, "prefill"),
        ([1024, 2048], 512, "prefill"),
        ([1, 512], 512, "mixed"),
        ([1, 1, 256], 512, "short"),  # max_q below min_q short-circuits first
        ([1] * 63 + [2048], 512, "mixed"),  # serve-shaped decode + one chunk
        ([128, 256], 512, "short"),
        ([600, 400], 512, "mixed"),  # cheap prefilter: total < batch * min_q
        ([4096, 300], 512, "short"),  # exact path: min_len 300 < min_q
        ([2048, 2048], 0, "prefill"),
        ([1, 2048], 0, "mixed"),  # min_q disabled still rejects mixed
    ],
)
def test_classify_prefill_batch(q_lens, min_q, expected):
    lens = torch.tensor(q_lens, dtype=torch.int32)
    cu = torch.zeros(len(q_lens) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(lens, dim=0)
    max_q = int(lens.max())
    num_q_tokens = int(lens.sum())
    assert _classify_prefill_batch(cu, max_q, num_q_tokens, min_q) == expected


@pytest.mark.parametrize("convert", [_bndh_to_bhnd, _bndh_to_hnd])
def test_bndh_to_hnd_layout_is_a_zero_copy_view(convert):
    num_blocks, page_size, num_heads, head_size = 3, 16, 4, 32
    # Stride padding models cache allocations whose block axis is non-contiguous.
    storage = torch.arange(
        num_blocks * 2 * page_size * num_heads * head_size,
        dtype=torch.float32,
    ).reshape(num_blocks * 2, page_size, num_heads, head_size)
    cache = storage[::2]

    hnd = convert(cache)

    assert hnd.shape == (num_blocks, num_heads, page_size, head_size)
    assert hnd.stride(-1) == 1
    assert hnd.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert hnd.storage_offset() == cache.storage_offset()
    torch.testing.assert_close(hnd, cache.permute(0, 2, 1, 3))


def _ixformer_supports_cache(cache):
    query = torch.empty((2, 4, 64), dtype=torch.bfloat16)
    return _is_supported_ixformer_call(
        q=query,
        k=cache,
        v=cache,
        block_table=torch.empty((1, 8), dtype=torch.int32),
        softcap=0.0,
        q_descale=None,
        alibi_slopes=None,
        sinks=None,
        qq_bias=None,
        mm_prefix_range=None,
        output_scale=None,
        use_td=False,
        kv_quant_mode=0,
        k_scale_cache=None,
        v_scale_cache=None,
        chunk_lookback=-1,
        window_size=(-1, -1),
    )


def test_ixformer_supports_strided_pages_with_contiguous_head_dim():
    cache = torch.empty((2, 16, 2, 64), dtype=torch.bfloat16)
    assert _ixformer_supports_cache(cache)


def test_ixformer_rejects_noncontiguous_head_dim():
    cache = torch.empty((2, 16, 2, 128), dtype=torch.bfloat16)[..., ::2]
    assert cache.shape == (2, 16, 2, 64)
    assert not _ixformer_supports_cache(cache)


def test_prebuild_split_meta_stashes_pages_and_cu_for_mixed_batch():
    # 4 decode rows + one 8-token chunk.  Metadata must be reusable across the
    # 42 layers of a step, so the stash has to hold real tensors, not just counts.
    n_dec, chunk = 4, 8
    page_size, max_pages = 16, 4
    device = "cpu"
    bt = torch.arange( (n_dec + 1) * max_pages, dtype=torch.int32, device=device).reshape(
        n_dec + 1, max_pages
    )
    sk = torch.tensor([32, 48, 16, 64, 80], dtype=torch.int32, device=device)
    q_lens = torch.tensor([1] * n_dec + [chunk], dtype=torch.int32, device=device)
    cu = torch.zeros(n_dec + 2, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(q_lens, 0)

    prebuild_split_meta(
        n_dec, n_dec, int(q_lens.sum()), bt, sk, cu, page_size
    )
    assert _usable_batch_split(int(q_lens.sum()), n_dec + 1) == (n_dec, n_dec)
    pages = _cached_decode_pages(n_dec)
    assert pages is not None
    indptr, indices, last_page_len = pages
    assert indptr.shape == (n_dec + 1,)
    assert last_page_len.shape == (n_dec,)
    assert _BATCH_SPLIT["prefill_cu_k"] is not None
    assert _BATCH_SPLIT["prefill_cu_k"].tolist() == [0, 80]
    assert _BATCH_SPLIT["prefill_cu_q"].tolist() == [0, chunk]
    # Wrong sequence count must not reuse the decode pages.
    assert _cached_decode_pages(n_dec + 1) is None


def test_prebuild_split_meta_pure_decode_stashes_pages_only():
    n, page_size, max_pages = 8, 16, 2
    bt = torch.arange(n * max_pages, dtype=torch.int32).reshape(n, max_pages)
    sk = torch.full((n,), 24, dtype=torch.int32)
    cu = torch.arange(n + 1, dtype=torch.int32)
    prebuild_split_meta(n, n, n, bt, sk, cu, page_size)
    # Pure decode is not a "usable split" (no prefill half), but pages are there.
    assert _usable_batch_split(n, n) is None
    assert _cached_decode_pages(n) is not None
    assert _BATCH_SPLIT["prefill_cu_k"] is None
    assert _BATCH_SPLIT["prefill_cu_q"] is None


def _ixformer_available():
    try:
        import ixformer.inference.functions.paged_attention  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _ixformer_available(), reason="needs ixformer + GPU")
def test_decode_graph_replay_ignores_the_per_step_page_stash():
    # The builder republishes freshly allocated pages every step.  A FULL decode
    # graph that captured the stash would replay against freed (and possibly
    # reused) memory, so it must rebuild pages from block_table/seq_lens inside
    # the graph instead.
    from flag_gems.runtime.backend._iluvatar.fused.paged_attention_decode import (
        paged_attention_decode,
    )
    from flag_gems.runtime.backend._iluvatar.fused.ixformer_unified_attention import (
        try_ixformer_unified_attention,
    )

    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    batch, heads, kv_heads, dim, page_size, max_pages = 48, 16, 2, 128, 16, 192
    num_blocks = batch * max_pages + 1
    # HND storage seen through vLLM's BNDH view, as select_hnd_kv_cache_layout does
    k = torch.randn(num_blocks, kv_heads, page_size, dim, device=dev, dtype=dt)
    v = torch.randn_like(k)
    k, v = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    q = torch.randn(batch, heads, dim, device=dev, dtype=dt)
    out = torch.empty_like(q)
    bt = torch.randperm(num_blocks - 1, device=dev, dtype=torch.int32)[
        : batch * max_pages
    ].reshape(batch, max_pages)
    sk = torch.ones(batch, device=dev, dtype=torch.int32)
    cu = torch.arange(batch + 1, device=dev, dtype=torch.int32)
    scale = dim**-0.5

    def step():
        prebuild_split_meta(batch, batch, batch, bt, sk, cu, page_size)

    def attend():
        assert try_ixformer_unified_attention(
            q, k, v, out, cu, 1, sk, page_size * max_pages, scale, True,
            block_table=bt,
        )

    step()
    attend()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attend()

    for it in range(3):
        sk.copy_(torch.randint(1024, page_size * max_pages, (batch,), device=dev))
        bt.copy_(bt[torch.randperm(batch, device=dev)])
        step()
        # churn the allocator so freed stash blocks get new contents
        junk = [torch.full((batch * max_pages,), 10**6, device=dev, dtype=torch.int32)
                for _ in range(8)]
        graph.replay()
        ref = paged_attention_decode(
            torch.empty_like(q), q, k, v, bt, sk, scale,
            max_seqlen_k=page_size * max_pages,
        )
        torch.cuda.synchronize()
        del junk
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)
