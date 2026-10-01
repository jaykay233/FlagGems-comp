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
    _is_supported_ixformer_call,
)


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
