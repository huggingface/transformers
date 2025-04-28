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

    Note that the default parameters listed here are intended as sensible defaults for the `convaiinnovations/hindi-causal-lm`
    model. They may not be optimal for all use cases or fine-tuned versions.

    Args:
        max_length (`int`, *optional*, defaults to 128):
            The maximum length of the sequence to be generated.
        temperature (`float`, *optional*, defaults to 0.7):
            The value used to modulate the next token probabilities. Higher values make the output more random, lower values make it more deterministic.
        top_k (`int`, *optional*, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 0.9):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
        do_sample (`bool`, *optional*, defaults to `True`):
            Whether or not to use sampling; use greedy decoding otherwise.
        pad_token_id (`int`, *optional*, defaults to 0):
             The id of the *padding* token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the *end-of-sequence* token.
        repetition_penalty (`float`, *optional*, defaults to 1.1):
            The parameter for repetition penalty. 1.0 means no penalty.
        kwargs:
            Additional generation parameters.
    """

    def __init__(
        self,
        max_length=128,
        temperature=0.7,  # Slightly lower default temp
        top_k=50,
        top_p=0.9,
        do_sample=True,
        pad_token_id=0,  # Default based on provided config
        bos_token_id=1,  # Default based on provided config
        eos_token_id=2,  # Default based on provided config
        repetition_penalty=1.1,
        **kwargs,
    ):
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
