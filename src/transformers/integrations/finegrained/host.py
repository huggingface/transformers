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
"""Conversion op of `FineGrainedHfQuantizer`."""

from ...core_model_loading import ConversionOps


class MemberQuantize(ConversionOps):
    """Quantize a tensor with the quantize op of the member that owns its module.

    The loader takes the op before it reads the tensor, so the op finds the member when it runs.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, *, full_layer_name: str, **kwargs):
        member = self.hf_quantizer.member_for(full_layer_name.rpartition(".")[0])
        return member.get_quantize_ops().convert(input_dict, full_layer_name=full_layer_name, **kwargs)
