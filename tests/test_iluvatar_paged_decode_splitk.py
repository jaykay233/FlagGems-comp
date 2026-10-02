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

from flag_gems.runtime.backend._iluvatar.fused.paged_decode_splitk import (
    _WORKSPACE,
    choose_num_splits,
    paged_decode_splitk,
)

PAGE = 16
HEAD_DIM = 128
# MiniCPM5-2B: 16 query heads over 2 KV heads, in all 42 layers.
N_QH, N_KVH = 16, 2

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _make_batch(num_seqs, ctx, seed=0, ragged=True, dtype=torch.bfloat16):
    """Paged KV in HND storage, handed over as the BNDH view vLLM passes."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    lens = torch.full((num_seqs,), ctx, device="cuda", dtype=torch.int32)
    if ragged and ctx > 8:
        lens -= torch.randint(0, ctx // 8, (num_seqs,), device="cuda", generator=g).to(
            torch.int32
        )
        lens[0] = ctx
    nb = (ctx + PAGE - 1) // PAGE
    pool = num_seqs * nb + 8
    k_hnd = torch.randn(
        pool, N_KVH, PAGE, HEAD_DIM, dtype=dtype, device="cuda", generator=g
    )
    v_hnd = torch.randn(
        pool, N_KVH, PAGE, HEAD_DIM, dtype=dtype, device="cuda", generator=g
    )
    # Shuffled pages: the kernel must follow the block table, not assume
    # sequences own contiguous runs.
    perm = torch.randperm(pool, device="cuda", generator=g)[: num_seqs * nb]
    block_table = perm.reshape(num_seqs, nb).to(torch.int32).contiguous()
    q = torch.randn(
        num_seqs, N_QH, HEAD_DIM, dtype=dtype, device="cuda", generator=g
    )
    return q, k_hnd.permute(0, 2, 1, 3), v_hnd.permute(0, 2, 1, 3), block_table, lens


def _reference(q, k_cache, v_cache, block_table, lens, scale):
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    group = q.shape[1] // N_KVH
    for i in range(q.shape[0]):
        length = int(lens[i])
        pages = block_table[i, : (length + PAGE - 1) // PAGE].long()
        k = k_cache[pages].reshape(-1, N_KVH, HEAD_DIM)[:length].float()
        v = v_cache[pages].reshape(-1, N_KVH, HEAD_DIM)[:length].float()
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        p = torch.softmax(torch.einsum("hd,lhd->hl", q[i].float(), k) * scale, dim=-1)
        out[i] = torch.einsum("hl,lhd->hd", p, v)
    return out


def _run(q, kc, vc, bt, lens, **kwargs):
    out = torch.empty_like(q)
    paged_decode_splitk(
        out, q, kc, vc, bt, lens, 1.0 / (HEAD_DIM**0.5), **kwargs
    )
    return out


@cuda_only
@pytest.mark.parametrize(
    "num_seqs,ctx",
    [
        (1, 2048),  # the 16k tail: a single sequence left running
        (6, 17408),
        (8, 4096),
        (31, 8192),  # just below the flashinfer boundary
        (4, 17),  # context shorter than one split's share
        (2, 16),  # exactly one page
        (3, 1),  # first decode step after a one-token prompt
    ],
)
def test_matches_reference_attention(num_seqs, ctx):
    q, kc, vc, bt, lens = _make_batch(num_seqs, ctx, seed=num_seqs * 31 + ctx)
    ref = _reference(q, kc, vc, bt, lens, 1.0 / (HEAD_DIM**0.5))
    out = _run(q, kc, vc, bt, lens)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert (out.float() - ref).abs().max().item() < 0.05


@cuda_only
@pytest.mark.parametrize("num_splits", [1, 2, 8, 64])
def test_split_count_does_not_change_the_result(num_splits):
    """Splitting is a reduction order, so every count must agree.

    More splits than keys leaves some programs with nothing to do; those must
    drop out of the reduction rather than contribute a zero row.
    """
    q, kc, vc, bt, lens = _make_batch(4, 1024, seed=7)
    ref = _reference(q, kc, vc, bt, lens, 1.0 / (HEAD_DIM**0.5))
    out = _run(q, kc, vc, bt, lens, num_splits=num_splits)
    torch.cuda.synchronize()
    assert (out.float() - ref).abs().max().item() < 0.05


@cuda_only
def test_ragged_lengths_do_not_leak_across_sequences():
    """One long sequence next to very short ones.

    A sequence's keys must stop at its own length: with an even split of a
    long range, most programs of a short sequence are empty.
    """
    q, kc, vc, bt, lens = _make_batch(8, 8192, seed=11)
    lens[1::2] = 3
    ref = _reference(q, kc, vc, bt, lens, 1.0 / (HEAD_DIM**0.5))
    out = _run(q, kc, vc, bt, lens)
    torch.cuda.synchronize()
    assert (out.float() - ref).abs().max().item() < 0.05


@cuda_only
def test_graph_replay_follows_the_lengths_set_after_capture():
    """vLLM captures FULL decode graphs with seq_lens filled to 1.

    The split count is chosen on the host and frozen into the graph, while the
    lengths are only written before replay, so the kernel has to read them on
    the device.  Replay must match an eager call at the real lengths.
    """
    q, kc, vc, bt, lens = _make_batch(8, 8192, seed=13)
    scale = 1.0 / (HEAD_DIM**0.5)
    ref = _reference(q, kc, vc, bt, lens, scale)

    lens_buf = torch.ones_like(lens)
    out = torch.empty_like(q)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        paged_decode_splitk(out, q, kc, vc, bt, lens_buf, scale)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_decode_splitk(out, q, kc, vc, bt, lens_buf, scale)

    lens_buf.copy_(lens)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert (out.float() - ref).abs().max().item() < 0.05


@cuda_only
def test_graph_replay_survives_allocator_churn():
    """Partial buffers are reused across calls, so they must stay alive.

    A graph that captured them keeps replaying into the same addresses; if the
    cache ever handed those blocks back, other tensors would land on top of
    them (FINDINGS §17.13).
    """
    q, kc, vc, bt, lens = _make_batch(8, 4096, seed=17)
    scale = 1.0 / (HEAD_DIM**0.5)
    ref = _reference(q, kc, vc, bt, lens, scale)

    lens_buf = torch.ones_like(lens)
    out = torch.empty_like(q)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        paged_decode_splitk(out, q, kc, vc, bt, lens_buf, scale)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_decode_splitk(out, q, kc, vc, bt, lens_buf, scale)

    lens_buf.copy_(lens)
    for _ in range(4):
        junk = torch.randn(4 * 1024 * 1024, device="cuda")
        del junk
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert (out.float() - ref).abs().max().item() < 0.05


@cuda_only
def test_workspace_is_reused_per_shape():
    q, kc, vc, bt, lens = _make_batch(8, 1024, seed=19)
    _run(q, kc, vc, bt, lens)
    key = next(k for k in _WORKSPACE if k[0] == 8)
    first = _WORKSPACE[key][0].data_ptr()
    _run(q, kc, vc, bt, lens)
    assert _WORKSPACE[key][0].data_ptr() == first


def test_split_count_depends_only_on_the_batch_shape():
    """It cannot depend on the context: capture sees seq_lens filled to 1."""
    for num_seqs in (1, 4, 8, 16, 31, 64):
        assert choose_num_splits(num_seqs, N_KVH) == choose_num_splits(num_seqs, N_KVH)
        assert choose_num_splits(num_seqs, N_KVH) >= 1
    # Small batches are split hardest, since that is where the card idles.
    assert choose_num_splits(1, N_KVH) > choose_num_splits(31, N_KVH)
