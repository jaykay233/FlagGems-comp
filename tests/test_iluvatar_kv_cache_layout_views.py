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
    _classify_prefill_batch,
    _decode_is_unsafe,
    _is_supported_ixformer_call,
    _usable_batch_split,
    clear_batch_split,
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
