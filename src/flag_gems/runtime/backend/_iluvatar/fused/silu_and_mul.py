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

"""Iluvatar fused ``silu(gate) * up`` for decode-sized activations.

The generic ``flag_gems.fused.silu_and_mul`` goes through ``pointwise_dynamic``
+ ``libentry``, which on BI-V150 costs ~20 µs/call inside a CUDA graph for
MiniCPM5-2B's ``(64, 6144)`` halves (eager share looked like 11% of the step;
graph reality is ~0.8 ms ×42).  A single contiguous Triton kernel lands at
~5.6 µs (422 GB/s of a 4 µs BW floor) — see FINDINGS-iluvatar §17.54.

Only the contiguous bf16/fp16 decode path is specialized; everything else
falls back to ATen so prefill / odd layouts stay correct.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _fast_enabled() -> bool:
    # Default on (§17.54). FLAGGEMS_ILUVATAR_SILU_FAST=0 falls back to ATen
    # for A/B against the previous pointwise_dynamic path's effective cost.
    return os.getenv("FLAGGEMS_ILUVATAR_SILU_FAST", "1") != "0"


@triton.jit
def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
    # silu(g) = g * sigmoid(g); sigmoid via exp is what pointwise_dynamic used.
    s = g * tl.sigmoid(g)
    tl.store(out_ptr + offs, (s * u).to(out_ptr.dtype.element_ty), mask=mask)


def _fast_ok(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype not in (torch.bfloat16, torch.float16):
        return False
    if a.shape != b.shape or a.device != b.device:
        return False
    if not a.is_cuda or not b.is_cuda:
        return False
    return a.is_contiguous() and b.is_contiguous()


def _launch(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    n = a.numel()
    # BLOCK=512 / 4 warps was the graph minimum on BI-V150 for N=64*6144.
    block = 512 if n >= 512 else max(64, triton.next_power_of_2(n))
    grid = (triton.cdiv(n, block),)
    _silu_mul_kernel[grid](a, b, out, n, BLOCK=block, num_warps=4)
    return out


def silu_and_mul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not _fast_enabled() or not _fast_ok(A, B):
        return F.silu(A) * B
    out = torch.empty_like(A)
    return _launch(A.view(-1), B.view(-1), out.view(-1)).view_as(A)


def silu_and_mul_out(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if (
        not _fast_enabled()
        or not _fast_ok(A, B)
        or not out.is_contiguous()
        or out.shape != A.shape
    ):
        out.copy_(F.silu(A) * B)
        return out
    return _launch(A.view(-1), B.view(-1), out.view(-1)).view_as(out)
