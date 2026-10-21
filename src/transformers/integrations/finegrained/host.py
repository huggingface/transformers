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


class MemberDispatch(ConversionOps):
    """Run a module's tensors through the conversion of the member that owns it.

    Members that read the same keys share one converter. `conversions` maps each member to its own version. `None`
    maps to the model's own conversion, for the modules no member owns.
    """

    def __init__(self, hf_quantizer, conversions: dict):
        self.hf_quantizer = hf_quantizer
        self.conversions = conversions
        self.supports_many_to_many = any(
            op.supports_many_to_many for conversion in conversions.values() for op in conversion.operations
        )

    def convert(self, input_dict, *, full_layer_name: str, **kwargs):
        owner = self.hf_quantizer.member_for(full_layer_name.rpartition(".")[0])
        # a member that kept the model's conversion runs that one. A full-precision module without one runs any.
        fallback = self.conversions[None] if None in self.conversions else next(iter(self.conversions.values()))
        conversion = self.conversions.get(owner, fallback)
        tensors = {pattern: input_dict[pattern] for pattern in conversion.source_patterns if pattern in input_dict}
        kwargs.update(source_patterns=conversion.source_patterns, target_patterns=conversion.target_patterns)
        for op in conversion.operations:
            tensors = op.convert(tensors, full_layer_name=full_layer_name, **kwargs)
        return tensors

    @property
    def reverse_op(self) -> ConversionOps:
        reverse = {owner: conversion.reverse_transform() for owner, conversion in self.conversions.items()}
        return MemberDispatch(self.hf_quantizer, reverse)
