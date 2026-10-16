# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""The MXFP8 arm of the fine-grained quantizer."""

from .base import FineGrainedHfQuantizer


class FineGrainedMxfp8HfQuantizer(FineGrainedHfQuantizer):
    """MXFP8 ships the plain `weight` / `weight_scale_inv` pair, so it needs none of the key
    surgery the other producers do and adds nothing to the base. It is named all the same, so
    `AUTO_QUANTIZER_MAPPING` reads one class per key."""
