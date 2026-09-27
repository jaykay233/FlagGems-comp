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
"""Tuner key strategies for the Iluvatar (corex) backend.

Why this module exists
----------------------
``LibTuner`` buckets autotune keys so nearby shapes share one measured config
(``ConfigCache.get_key`` applies ``self.strategy``).  ``DEFAULT_STRATEGIES`` in
``flag_gems/runtime/common.py`` declares ``"mm": ["align32", ...]``, but that
table is only consumed for expand specs, i.e. under
``TuningMode.EXPANDED``.  A decorator that passes no ``strategy=`` and runs in
``TuningMode.DEFAULT`` therefore falls back to an *identity* key: the raw ``M``.

That is fine for N and K, which are static model dimensions, and wrong for M,
which vLLM drives from the data: chunked prefill hands the kernel an arbitrary
token count on every step.  Measured on this box, before this module existed,
one ``TritonConfigCache`` table held **134 distinct M values** for the bf16
``mm`` kernel, including runs like ``... 1909, 1911, 2030, 2032, 2034 ...`` -
adjacent, unaligned, and never repeating.  Every cold key costs a full autotune
(many Triton compiles plus timed benchmarks), and it is paid *on the inference
hot path*: the symptom is a TTFT distribution with a healthy median and a very
heavy tail (``Median 3.4 s`` vs ``P99 82 s``), which is exactly what the
organiser's mean-TTFT gate punishes.

``align32`` alone is not the answer: it is unbounded, so sweeping 1..2048 still
yields 69 keys.

``align32_geometric`` keeps ``align32``'s 32-granularity up to
``ALIGN32_GEOMETRIC_THRESHOLD`` - which preserves resolution across the decode /
small-batch region vLLM captures CUDA graphs for, where the tile choice is most
sensitive - then switches to geometric (powers of two) buckets.  The key space
becomes O(log M): 13 buckets over 1..2048 instead of 69, namely
``1 2 4 8 16 32 64 96 128 256 512 1024 2048``.

Only the *key* is coarsened; the kernel still receives the real M.  A config
chosen for a coarse bucket is applied to a range of nearby M values, which is
what ``align32`` already does inside its 32-wide windows.

This is a pure operator-compilation change: it does not alter which operator
runs, nor which backend implements it.
"""

import math
from typing import Union

from flag_gems.utils.libentry import LibTuner

# Above this, switch from 32-granularity to geometric buckets.
# 128 keeps fine resolution across the decode / small-prefill region (the batch
# sizes vLLM actually captures CUDA graphs for) while collapsing the unbounded
# chunked-prefill tail.  Raising it trades more autotunes for finer tile choice.
ALIGN32_GEOMETRIC_THRESHOLD = 128


def align32_geometric(key: Union[int, float]) -> int:
    """Bounded variant of ``align32`` for runtime-varying dimensions.

    * ``key == 0``                -> 0
    * ``key < 32``                -> next power of two (same as ``align32``)
    * ``32 <= key <= T``          -> round up to a multiple of 32 (same as ``align32``)
    * ``key > T``                 -> next power of two (bounds the key space)

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
