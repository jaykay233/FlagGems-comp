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

"""MetaX-specific ``linear`` implementation.

Why this file exists
--------------------
The generic ``flag_gems.ops.linear.linear`` launches ``linear_kernel``, a plain
blocked GEMM:

* ``pid_m`` / ``pid_n`` are taken straight from ``tl.program_id`` with no
  grouped (L2-swizzled) tile ordering;
* there is no entry for ``linear`` in ``_metax/tune_configs.yaml``, so the
  autotuner only sees the architecture-agnostic config list;
* the grid is ``cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)``, which for the decode
  shape (M == 1) collapses to a handful of programs and leaves the device idle;
* every K iteration reloads a fully masked block, so there is no pipelining.

Meanwhile ``_metax/ops/mm.py`` already ships a shape-aware, MetaX-tuned MM
family: cpasync pipelines, tile pruning against the L2 capacity, split-K for
low-occupancy shapes, a dedicated GEMV path, and ``mm_kernel_nt`` with
``GROUP_M`` tile ordering for transposed-weight (``nt``) layouts.

``linear(x, W)`` is ``mm(x, W.T)`` and ``W.T`` has strides ``(1, K)`` — exactly
the ``nt`` layout those kernels are tuned for. So rather than maintaining a
second, independently tuned GEMM, route ``linear`` through the MM family.

Layout mapping
--------------
``linear``  : x (M, K) contiguous, W (N, K) contiguous, y = x @ W.T + b
``mm(x, W.T)``: a = x with strides (K, 1), b = W.T with strides (1, K)
``nt_mm_scenario`` requires ``a.stride(0) == K``, ``a.stride(1) == 1``,
``b.stride(0) == 1``, ``b.stride(1) == K`` -> matches, dispatching to
``general_mm_nt`` (prefill) or ``splitk_mm`` (decode).

``mm()`` forwards both operands' strides into every kernel it launches, so the
``nt`` layout is handled correctly across all of its scenarios.
"""

import logging

import torch

from .mm import mm

logger = logging.getLogger(__name__)

__all__ = ["linear"]


def linear(input, weight, bias=None):
    """Applies ``y = x @ weight.T + bias`` using the MetaX-tuned MM family.

    Args:
        input: Input tensor of shape ``(*, in_features)``.
        weight: Weight tensor of shape ``(out_features, in_features)``.
        bias: Optional bias of shape ``(out_features,)``.

    Returns:
        Output tensor of shape ``(*, out_features)``.
    """
    logger.debug("GEMS_METAX LINEAR")

    input_dim = input.dim()
    single_1d = input_dim == 1
    if single_1d:
        # Treat a 1-D input as a single row.
        input = input.unsqueeze(0)

    batch_dims = input.shape[:-1]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim

    K = input.shape[-1]
    N = weight.shape[0]

    x = input.reshape(batch_size, K)
    if not x.is_contiguous():
        x = x.contiguous()

    if weight.dim() != 2:
        raise AssertionError(
            f"linear expects a 2-D weight, got shape {tuple(weight.shape)}"
        )
    if weight.shape[1] != K:
        raise AssertionError(
            f"linear got mismatched shapes: input {tuple(input.shape)} "
            f"and weight {tuple(weight.shape)}"
        )
    # Only normalise genuine odd strides; vLLM weights are contiguous, so this
    # is a no-op on the hot path and never copies per call.
    if not weight.is_contiguous():
        weight = weight.contiguous()

    # linear(x, W) == mm(x, W.T). W.T is a free view with strides (1, K), which
    # is exactly the `nt` layout mm_kernel_nt / split-K are tuned for.
    out = mm(x, weight.t())

    if bias is not None:
        out = out + bias

    if single_1d:
        return out.squeeze(0)
    if batch_dims:
        return out.view(*batch_dims, N)
    return out
