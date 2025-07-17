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

from ...utils import is_sentencepiece_available, logging
from ..llama import LlamaTokenizerFast


if is_sentencepiece_available():
    from .tokenization_ernie4_5 import Ernie4_5Tokenizer
else:
    Ernie4_5Tokenizer = None


logger = logging.get_logger(__name__)


DEFAULT_CHAT_TEMPLATE = '{%- if not add_generation_prompt is defined -%}\n    {%- set add_generation_prompt = true -%}\n{%- endif -%}\n{%- if not cls_token is defined -%}\n    {%- set cls_token = "<|begin_of_sentence|>" -%}\n{%- endif -%}\n{%- if not sep_token is defined -%}\n    {%- set sep_token = "<|end_of_sentence|>" -%}\n{%- endif -%}\n{{- cls_token -}}\n{%- for message in messages -%}\n    {%- if message["role"] == "user" -%}\n        {{- "User: " + message["content"] + "\n" -}}\n    {%- elif message["role"] == "assistant" -%}\n        {{- "Assistant: " + message["content"] + sep_token -}}\n    {%- elif message["role"] == "system" -%}\n        {{- message["content"] + "\n" -}}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{- "Assistant: " -}}\n{%- endif -%}'


class Ernie4_5TokenizerFast(LlamaTokenizerFast):
    # NOTE: Wrapper around Llama to enable different default values and warn
    # if unintended behavior might be encounters, e.g. `add_prefix_space`.
    #
    # See `Ernie4_5Tokenizer` for why these classes exist in particular.

    legacy = True
    add_prefix_space = False
    slow_tokenizer_class = Ernie4_5Tokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        pad_token="<unk>",
        cls_token="<|begin_of_sentence|>",
        sep_token="<|end_of_sentence|>",
        mask_token="<mask:1>",
        add_bos_token=False,
        add_prefix_space=False,
        chat_template=DEFAULT_CHAT_TEMPLATE,
        legacy=True,
        **kwargs,
    ):
        if add_prefix_space:
            logger.warning_once(
                "The Ernie 4.5 Tokenizer does not support `add_prefix_space` so it may lead to unexpected behavior between "
                "the slow and fast tokenizer versions."
            )

        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            add_bos_token=add_bos_token,
            add_prefix_space=add_prefix_space,
            chat_template=chat_template,
            legacy=legacy,
            **kwargs,
        )


__all__ = ["Ernie4_5TokenizerFast"]
