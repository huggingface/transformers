# Copyright (c) 2025 Baidu, Inc. and HuggingFace Inc. team. All Rights Reserved.
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
"""Tokenization classes for Ernie 4.5"""

from ...utils import is_sentencepiece_available
from ..llama import LlamaTokenizerFast


if is_sentencepiece_available():
    from .tokenization_ernie4_5 import Ernie4_5Tokenizer
else:
    Ernie4_5Tokenizer = None


DEFAULT_CHAT_TEMPLATE = '{%- if not add_generation_prompt is defined -%}\n    {%- set add_generation_prompt = true -%}\n{%- endif -%}\n{%- if not cls_token is defined -%}\n    {%- set cls_token = "<|begin_of_sentence|>" -%}\n{%- endif -%}\n{%- if not sep_token is defined -%}\n    {%- set sep_token = "<|end_of_sentence|>" -%}\n{%- endif -%}\n{{- cls_token -}}\n{%- for message in messages -%}\n    {%- if message["role"] == "user" -%}\n        {{- "User: " + message["content"] + "\n" -}}\n    {%- elif message["role"] == "assistant" -%}\n        {{- "Assistant: " + message["content"] + sep_token -}}\n    {%- elif message["role"] == "system" -%}\n        {{- message["content"] + "\n" -}}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{- "Assistant: " -}}\n{%- endif -%}'


class Ernie4_5TokenizerFast(LlamaTokenizerFast):
    legacy = False
    add_prefix_space = False
    slow_tokenizer_class = Ernie4_5Tokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<unk>",
        unk_token="<unk>",
        cls_token="<|begin_of_sentence|>",
        sep_token="<|end_of_sentence|>",
        mask_token="<mask:1>",
        add_bos_token=False,
        add_eos_token=False,
        chat_template=DEFAULT_CHAT_TEMPLATE,
        clean_up_tokenization_spaces=False,
        spaces_between_special_tokens=False,
        add_prefix_space=False,
        legacy=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            chat_template=chat_template,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            add_prefix_space=add_prefix_space,
            legacy=legacy,
            **kwargs,
        )


__all__ = ["Ernie4_5TokenizerFast"]
