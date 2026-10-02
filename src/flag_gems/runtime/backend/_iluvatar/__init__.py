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

from flag_gems.runtime.backend.backend_utils import VendorDescriptor


def get_triton_extra_name():
    try:
        import triton
        from packaging import version

        if version.parse(triton.__version__) >= version.parse("3.6.0"):
            return "corex"
        return "cuda"
    except Exception:
        return "cuda"


vendor_info = VendorDescriptor(
    vendor_name="iluvatar",
    device_name="cuda",
    device_query_cmd="ixsmi",
    triton_extra_name=get_triton_extra_name(),
    fp64_enabled=False,
)

# Small eager-path ops (residual add, attention/sampler metadata) stay on native
# aten.  On BI-V150 a FlagGems Triton pointwise launch costs ~0.3 ms of host vs
# ~0.01 ms native, and several of these JIT a new kernel per input shape or
# specialization (``index`` keys its code cache on the full input shape), which
# stalls the first serving pass for tens of seconds on a cold Triton cache.
CUSTOMIZED_UNUSED_OPS = (
    "add",
    "sub",
    "index",
    "copy_",
    "to_copy",
    "eq",
    "eq_scalar",
    "floor_divide",
    "remainder",
    "where_self",
    "cumsum",
    "full",
    "full_like",
    "fill_scalar_",
    "zeros",
    "zeros_like",
    "zero_",
)

__all__ = ["*"]
