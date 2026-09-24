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
"""Tuner key strategies used by the MetaX backend.

Why this module exists
----------------------
``LibTuner`` can bucket autotune keys so nearby shapes share one measured
config (``ConfigCache.get_key`` applies ``self.strategy``). ``align32`` is the
strategy declared for the GEMM ops in
``flag_gems/runtime/common.py::DEFAULT_STRATEGIES``, and it is a good choice for
*static* dimensions such as N or K.

It is a poor choice for a *runtime* dimension. ``align32`` returns
``ceil(key / 32) * 32`` with no upper bound, so a batch dimension that sweeps
1..2048 produces 69 distinct keys - and each cold key costs one full autotune.
Measured on MetaX C500, a single ``mm`` autotune is 7-9 s (8 configs, each
Triton-compiled and then benchmarked under ``BenchmarkMode.REPLAY``), so the
worst case for one kernel is ~10 minutes of tuning *on the inference hot path*.
``vLLM`` hits exactly this: the prefill token count is data-dependent, so the
key space never warms up.

``align32_geometric`` keeps ``align32``'s 32-granularity up to a threshold -
preserving resolution in the decode / small-batch region where the tile choice
matters most - and switches to geometric (powers of two) buckets above it. The
key space becomes O(log M): 13 buckets instead of 69 across 1..2048, i.e.
~110 s instead of ~586 s worst case, with the buckets
``1 2 4 8 16 32 64 96 128 256 512 1024 2048``.

Only the *key* is coarsened; the kernel still sees the real M. A config chosen
for a coarser bucket is applied to a range of nearby M values, which is exactly
what ``align32`` already does within its 32-wide windows.
"""

import math
from typing import Union

from flag_gems.utils.libentry import LibTuner

# Above this, switch from 32-granularity to geometric buckets.
# 128 keeps fine resolution across the decode / small-prefill region
# (the batch sizes vLLM actually captures CUDA graphs for) while collapsing
# the long prefill tail.
ALIGN32_GEOMETRIC_THRESHOLD = 128


def align32_geometric(key: Union[int, float]) -> int:
    """Bounded variant of ``align32`` for runtime-varying dimensions.

    * ``key == 0``        -> 0
    * ``key < 32``        -> next power of two (same as ``align32``)
    * ``32 <= key <= T``  -> round up to a multiple of 32 (same as ``align32``)
    * ``key > T``         -> next power of two (bounds the key space)

    >>> [align32_geometric(k) for k in (1, 33, 100, 128, 1500, 2048)]
    [1, 64, 128, 128, 2048, 2048]
    """
    if key == 0:
        return 0
    if key < 32:
        return 2 ** math.ceil(math.log2(key))
    if key <= ALIGN32_GEOMETRIC_THRESHOLD:
        return math.ceil(key / 32) * 32
    return 2 ** math.ceil(math.log2(key))


LibTuner.register_strategy("align32_geometric")(align32_geometric)

__all__ = ["ALIGN32_GEOMETRIC_THRESHOLD", "align32_geometric"]
