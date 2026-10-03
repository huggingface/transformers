# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""The fine-grained quantization integration: `core` holds the modules and their forwards,
`conversions` the loader ops that fill them. `conversions` is imported by path, so that reading
a checkpoint's layout never drags the kernels in."""

from .core import (
    ALL_FINEGRAINED_EXPERTS_FUNCTIONS,
    FineGrainedEmbedding,
    FineGrainedExperts,
    FineGrainedGroupedLinear,
    FineGrainedLinear,
    assert_modules_are_quantized,
    disable_deepgemm_on_multi_device,
    finegrained_linear,
    load_finegrained_kernel,
    replace_with_finegrained_embedding,
    replace_with_finegrained_layer,
)


__all__ = [
    "ALL_FINEGRAINED_EXPERTS_FUNCTIONS",
    "FineGrainedEmbedding",
    "FineGrainedExperts",
    "FineGrainedGroupedLinear",
    "FineGrainedLinear",
    "assert_modules_are_quantized",
    "disable_deepgemm_on_multi_device",
    "finegrained_linear",
    "load_finegrained_kernel",
    "replace_with_finegrained_embedding",
    "replace_with_finegrained_layer",
]
