# coding=utf-8
# Copyright 2024 ConvaiInnovations and The HuggingFace Inc. team. All rights reserved.
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
"""
Default generation configuration for HindiCausalLM.
"""

from ...generation.configuration_utils import GenerationConfig


class HindiCausalLMGenerationConfig(GenerationConfig):
    """
    This class contains the default [`~generation.GenerationConfig`] for the [`HindiCausalLMForCausalLM`] model. See
    [`~generation.GenerationConfig`] for all details regarding parameters.
    """

    def __init__(
        self,
        max_length=128,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        repetition_penalty=1.1,
        **kwargs,
    ):
        """
        Initialize with a set of parameters that work well for generating Hindi text with HindiCausalLM.
        """
        super().__init__(
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
