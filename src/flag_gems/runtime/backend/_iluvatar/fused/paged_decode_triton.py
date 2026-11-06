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

"""Triton FlashDecoding decode backend for BI-V150.

One program per ``(sequence, kv_head, context split)``: it walks its slice of
the paged KV cache, keeps a running unnormalised ``sum(exp * v)`` and
``sum(exp)``, and a second lightweight kernel folds the splits back together.

Two K layouts are supported:

* **BNDH** ``[blocks, page, n_kv, hd]`` (vLLM unified_attention view): K is
  loaded row-major ``[page, hd]``, scores via ``tl.trans(tl.dot(K, Q^T))``.
  Microbench ~0.488 ms — matches warm FA (~0.487 ms).
* **K-contig / HD-major** ``[blocks, n_kv, hd, page]``: K loads as the MMA
  ``[hd, page]`` operand with full coalescing.  Microbench ~0.465 ms —
  **beats FA by ~20 µs** at B=64 CTX=4600.  V stays BNDH.

Also: log2-domain softmax (no per-page rescale); hot loop unmasked except the
final partial page.

Default ON (``VLLM_FL_ILUVATAR_DECODE_TRITON=1``); set ``=0`` for vendor FA.
See FINDINGS-iluvatar.
"""


from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# Programs per (sequence, kv_head) are chosen so the grid covers all 16 SMs
# with enough CTAs to hide latency; past ~256 CTAs a wider split only adds
# combine traffic.
_CTA_TARGET = 256
_SPLIT_MIN = 2
_SPLIT_MAX = 16
# Fewer pages per split than this is pure overhead.
_MIN_PAGES_PER_SPLIT = 8
# exp2 argument cap: 2**80 is 1.2e24, so a few thousand of them stay inside
# fp32 and the ratio is unaffected by clamping (a logit that large has already
# saturated the softmax).
_LOGIT_CLAMP = tl.constexpr(80.0)

_WORKSPACE: dict = {}


def _enabled() -> bool:
    return os.getenv("VLLM_FL_ILUVATAR_DECODE_TRITON", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


@triton.jit
def _paged_decode_kernel(
    q_ptr, k_ptr, v_ptr, bt_ptr, sl_ptr, o_ptr, l_ptr,
    stride_qb, stride_qh,
    stride_kb, stride_kp, stride_kh,
    stride_vb, stride_vp, stride_vh,
    stride_btb, stride_os,
    scale,
    N_QO: tl.constexpr, N_KV: tl.constexpr, HD: tl.constexpr,
    PAGE: tl.constexpr, BM: tl.constexpr, GROUP: tl.constexpr,
    SPLIT: tl.constexpr, OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    split = pid % SPLIT
    rest = pid // SPLIT
    kvh = rest % N_KV
    b = rest // N_KV

    seqlen = tl.load(sl_ptr + b)
    npages = (seqlen + PAGE - 1) // PAGE
    chunk = (npages + SPLIT - 1) // SPLIT
    p_start = split * chunk
    p_end = tl.minimum(p_start + chunk, npages)
    # Only the last page of a sequence can be partial, so the hot loop needs
    # no masks at all and the masked copy runs at most once.  The outer
    # maximum matters: a split with no pages left would otherwise clamp to a
    # full-page index *behind* its own start and the tail loop would re-count
    # a page another split already owns.
    p_full_end = tl.maximum(p_start, tl.minimum(p_end, seqlen // PAGE))

    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, HD)
    mask_m = offs_m < GROUP
    qh = kvh * GROUP + offs_m
    q = tl.load(
        q_ptr + b * stride_qb + qh[:, None] * stride_qh + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )
    # Right-operand Q so K can stay contiguous [PAGE, HD]; only this small
    # [HD, BM] tile (and the [PAGE, BM] scores below) need a transpose.
    qT = tl.trans(q)

    acc = tl.zeros((BM, HD), dtype=tl.float32)
    l_i = tl.zeros((BM,), dtype=tl.float32)
    bt_base = bt_ptr + b * stride_btb
    slot = tl.arange(0, PAGE)

    for p in range(p_start, p_full_end):
        page_id = tl.load(bt_base + p)
        kb = k_ptr + page_id * stride_kb + kvh * stride_kh + slot * stride_kp
        vb = v_ptr + page_id * stride_vb + kvh * stride_vh + slot * stride_vp
        k_tile = tl.load(kb[:, None] + offs_d[None, :], cache_modifier=".cg")
        v_tile = tl.load(vb[:, None] + offs_d[None, :], cache_modifier=".cg")
        s = tl.trans(tl.dot(k_tile, qT)) * scale
        p_ = tl.exp2(tl.minimum(s, _LOGIT_CLAMP))
        l_i += tl.sum(p_, 1)
        acc = tl.dot(p_.to(OUT_DTYPE), v_tile, acc)

    for p in range(p_full_end, p_end):
        page_id = tl.load(bt_base + p)
        valid = slot < (seqlen - p * PAGE)
        kb = k_ptr + page_id * stride_kb + kvh * stride_kh + slot * stride_kp
        vb = v_ptr + page_id * stride_vb + kvh * stride_vh + slot * stride_vp
        k_tile = tl.load(kb[:, None] + offs_d[None, :], mask=valid[:, None],
                         other=0.0)
        v_tile = tl.load(vb[:, None] + offs_d[None, :], mask=valid[:, None],
                         other=0.0)
        s = tl.trans(tl.dot(k_tile, qT)) * scale
        s = tl.where(valid[None, :], s, float("-inf"))
        p_ = tl.exp2(tl.minimum(s, _LOGIT_CLAMP))
        l_i += tl.sum(p_, 1)
        acc = tl.dot(p_.to(OUT_DTYPE), v_tile, acc)

    # With no combine pass the split has to normalise itself.
    if SPLIT == 1:
        acc = acc / l_i[:, None]

    o_off = (split * stride_os + b * N_QO * HD + qh[:, None] * HD
             + offs_d[None, :])
    tl.store(o_ptr + o_off, acc.to(OUT_DTYPE), mask=mask_m[:, None])
    tl.store(l_ptr + b * N_QO * SPLIT + qh * SPLIT + split, l_i, mask=mask_m)



@triton.jit
def _paged_decode_kernel_k_contig(
    q_ptr, k_ptr, v_ptr, bt_ptr, sl_ptr, o_ptr, l_ptr,
    stride_qb, stride_qh,
    stride_kb, stride_kd, stride_ks,
    stride_vb, stride_vp, stride_vh,
    stride_btb, stride_os,
    scale,
    N_QO: tl.constexpr, N_KV: tl.constexpr, HD: tl.constexpr,
    PAGE: tl.constexpr, BM: tl.constexpr, GROUP: tl.constexpr,
    SPLIT: tl.constexpr, OUT_DTYPE: tl.constexpr,
):
    """K is [blocks, n_kv, HD, PAGE] contiguous — ~465µs vs FA ~487µs at 4k/n=64."""
    pid = tl.program_id(0)
    split = pid % SPLIT
    rest = pid // SPLIT
    kvh = rest % N_KV
    b = rest // N_KV

    seqlen = tl.load(sl_ptr + b)
    npages = (seqlen + PAGE - 1) // PAGE
    chunk = (npages + SPLIT - 1) // SPLIT
    p_start = split * chunk
    p_end = tl.minimum(p_start + chunk, npages)
    p_full_end = tl.maximum(p_start, tl.minimum(p_end, seqlen // PAGE))

    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, HD)
    mask_m = offs_m < GROUP
    qh = kvh * GROUP + offs_m
    q = tl.load(
        q_ptr + b * stride_qb + qh[:, None] * stride_qh + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )

    acc = tl.zeros((BM, HD), dtype=tl.float32)
    l_i = tl.zeros((BM,), dtype=tl.float32)
    bt_base = bt_ptr + b * stride_btb
    slot = tl.arange(0, PAGE)

    for p in range(p_start, p_full_end):
        page_id = tl.load(bt_base + p)
        kb = (k_ptr + page_id * stride_kb + kvh * (HD * PAGE)
              + offs_d[:, None] * stride_kd + slot[None, :] * stride_ks)
        k_tile = tl.load(kb, cache_modifier=".cg")
        vb = v_ptr + page_id * stride_vb + kvh * stride_vh + slot * stride_vp
        v_tile = tl.load(vb[:, None] + offs_d[None, :], cache_modifier=".cg")
        s = tl.dot(q, k_tile) * scale
        p_ = tl.exp2(tl.minimum(s, _LOGIT_CLAMP))
        l_i += tl.sum(p_, 1)
        acc = tl.dot(p_.to(OUT_DTYPE), v_tile, acc)

    for p in range(p_full_end, p_end):
        page_id = tl.load(bt_base + p)
        valid = slot < (seqlen - p * PAGE)
        kb = (k_ptr + page_id * stride_kb + kvh * (HD * PAGE)
              + offs_d[:, None] * stride_kd + slot[None, :] * stride_ks)
        k_tile = tl.load(kb, mask=valid[None, :], other=0.0)
        vb = v_ptr + page_id * stride_vb + kvh * stride_vh + slot * stride_vp
        v_tile = tl.load(vb[:, None] + offs_d[None, :], mask=valid[:, None],
                         other=0.0)
        s = tl.dot(q, k_tile) * scale
        s = tl.where(valid[None, :], s, float("-inf"))
        p_ = tl.exp2(tl.minimum(s, _LOGIT_CLAMP))
        l_i += tl.sum(p_, 1)
        acc = tl.dot(p_.to(OUT_DTYPE), v_tile, acc)

    if SPLIT == 1:
        acc = acc / l_i[:, None]

    o_off = (split * stride_os + b * N_QO * HD + qh[:, None] * HD
             + offs_d[None, :])
    tl.store(o_ptr + o_off, acc.to(OUT_DTYPE), mask=mask_m[:, None])
    tl.store(l_ptr + b * N_QO * SPLIT + qh * SPLIT + split, l_i, mask=mask_m)


@triton.jit
def _paged_decode_combine(
    o_ptr, l_ptr, out_ptr, stride_os,
    N_QO: tl.constexpr, HD: tl.constexpr, SPLIT: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // N_QO
    qh = pid % N_QO
    offs_d = tl.arange(0, HD)
    offs_s = tl.arange(0, SPLIT)
    denom = tl.sum(tl.load(l_ptr + b * N_QO * SPLIT + qh * SPLIT + offs_s))
    acc = tl.zeros((HD,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(o_ptr + s * stride_os + b * N_QO * HD + qh * HD
                       + offs_d).to(tl.float32)
    tl.store(out_ptr + b * N_QO * HD + qh * HD + offs_d,
             (acc / denom).to(OUT_DTYPE))


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _k_layout(k: torch.Tensor, head_dim: int):
    """Return (kind, n_kv, page) for BNDH or HD-major K."""
    # HD-major: [blocks, n_kv, hd, page]
    if (k.shape[2] == head_dim and k.shape[3] in (16, 32, 64, 128)
            and k.shape[1] <= 64 and k.stride(3) == 1 and k.stride(2) == k.shape[3]):
        return "hd_major", int(k.shape[1]), int(k.shape[3])
    # BNDH: [blocks, page, n_kv, hd]
    if k.shape[3] == head_dim and k.shape[1] in (16, 32, 64, 128):
        return "bndh", int(k.shape[2]), int(k.shape[1])
    return None, 0, 0


def usable(q: torch.Tensor, k: torch.Tensor, out: torch.Tensor) -> bool:
    """Shape/dtype gate: exactly the things this kernel indexes directly."""
    head_dim = int(q.shape[-1])
    if not _is_pow2(head_dim) or not 16 <= head_dim <= 256:
        return False
    if k.dim() != 4 or k.stride(-1) != 1:
        return False
    kind, n_kv, _page = _k_layout(k, head_dim)
    if kind is None or n_kv == 0 or int(q.shape[1]) % n_kv:
        return False
    if int(q.shape[1]) // n_kv > 32:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False
    if k.dtype != q.dtype or out.dtype != q.dtype:
        return False
    if not (q.is_contiguous() and out.is_contiguous()):
        return False
    return True


def _pick_split(batch: int, n_kv: int, max_pages: int) -> int:
    split = _SPLIT_MIN
    want = -(-_CTA_TARGET // max(1, batch * n_kv))
    while split < want and split < _SPLIT_MAX:
        split *= 2
    while split > 1 and max_pages // split < _MIN_PAGES_PER_SPLIT:
        split //= 2
    return max(1, split)


def _workspace(split, batch, n_heads, head_dim, dtype, device):
    key = (str(device), split, batch, n_heads, head_dim, dtype)
    ws = _WORKSPACE.get(key)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            # Allocating under capture would bake a transient address into the
            # graph; fall back so the caller stays on a warmed path.
            return None
        ws = (
            torch.empty(split, batch, n_heads, head_dim, device=device,
                        dtype=dtype),
            torch.zeros(batch, n_heads, split, device=device,
                        dtype=torch.float32),
        )
        _WORKSPACE[key] = ws
    return ws


def paged_decode_triton(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
) -> bool:
    """Run the Triton FlashDecoding decode kernel.  ``True`` = wrote ``out``."""
    if not usable(query, key_cache, out):
        return False

    batch, n_qo, head_dim = (int(x) for x in query.shape)
    kind, n_kv, page = _k_layout(key_cache, head_dim)
    if kind is None:
        return False
    group = n_qo // n_kv
    bm = max(4, triton.next_power_of_2(group))
    split = _pick_split(batch, n_kv, int(block_table.shape[1]))

    ws = _workspace(split, batch, n_qo, head_dim, query.dtype, query.device)
    if ws is None:
        return False
    o_buf, l_buf = ws

    bt = block_table if block_table.dtype == torch.int32 \
        else block_table.to(torch.int32)
    sl = seqused_k if seqused_k.dtype == torch.int32 \
        else seqused_k.to(torch.int32)
    if not bt.is_contiguous():
        bt = bt.contiguous()
    if not sl.is_contiguous():
        sl = sl.contiguous()

    # Softmax in the log2 domain: fold log2(e) into the scale once.
    scale = float(softmax_scale) * 1.4426950408889634
    out_dtype = tl.bfloat16 if query.dtype == torch.bfloat16 else tl.float16

    if split == 1:
        # No combine pass: accumulate straight into the caller's buffer.
        o_buf = out.view(1, *out.shape)
        l_buf = l_buf[:, :, :1]

    if kind == "hd_major":
        _paged_decode_kernel_k_contig[(batch * n_kv * split,)](
            query, key_cache, value_cache, bt, sl, o_buf, l_buf,
            query.stride(0), query.stride(1),
            key_cache.stride(0), key_cache.stride(2), key_cache.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            bt.stride(0), o_buf.stride(0), scale,
            N_QO=n_qo, N_KV=n_kv, HD=head_dim, PAGE=page, BM=bm, GROUP=group,
            SPLIT=split, OUT_DTYPE=out_dtype,
            num_warps=1, num_stages=1,
        )
    else:
        _paged_decode_kernel[(batch * n_kv * split,)](
            query, key_cache, value_cache, bt, sl, o_buf, l_buf,
            query.stride(0), query.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            bt.stride(0), o_buf.stride(0), scale,
            N_QO=n_qo, N_KV=n_kv, HD=head_dim, PAGE=page, BM=bm, GROUP=group,
            SPLIT=split, OUT_DTYPE=out_dtype,
            num_warps=1, num_stages=1,
        )
    if split > 1:
        _paged_decode_combine[(batch * n_qo,)](
            o_buf, l_buf, out, o_buf.stride(0),
            N_QO=n_qo, HD=head_dim, SPLIT=split, OUT_DTYPE=out_dtype,
            num_warps=1,
        )
    return True
