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

from typing import Any, Optional

from ...tokenization_utils import AddedToken
from ...utils import logging
from ...utils.import_utils import requires
from ..llama import LlamaTokenizer


logger = logging.get_logger(__name__)


DEFAULT_CHAT_TEMPLATE = '{%- if not add_generation_prompt is defined -%}\n    {%- set add_generation_prompt = true -%}\n{%- endif -%}\n{%- if not cls_token is defined -%}\n    {%- set cls_token = "<|begin_of_sentence|>" -%}\n{%- endif -%}\n{%- if not sep_token is defined -%}\n    {%- set sep_token = "<|end_of_sentence|>" -%}\n{%- endif -%}\n{{- cls_token -}}\n{%- for message in messages -%}\n    {%- if message["role"] == "user" -%}\n        {{- "User: " + message["content"] + "\n" -}}\n    {%- elif message["role"] == "assistant" -%}\n        {{- "Assistant: " + message["content"] + sep_token -}}\n    {%- elif message["role"] == "system" -%}\n        {{- message["content"] + "\n" -}}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{- "Assistant: " -}}\n{%- endif -%}'


@requires(backends=("sentencepiece",))
class Ernie4_5Tokenizer(LlamaTokenizer):
    legacy = False
    padding_side = "left"
    add_prefix_space = False

    def __init__(
        self,
        vocab_file,
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
        sp_model_kwargs: Optional[dict[str, Any]] = None,
        clean_up_tokenization_spaces=False,
        spaces_between_special_tokens=False,
        add_prefix_space=False,
        legacy=False,
        **kwargs,
    ):
        cls_token = AddedToken(cls_token, normalized=False, special=True) if isinstance(cls_token, str) else cls_token
        sep_token = AddedToken(sep_token, normalized=False, special=True) if isinstance(sep_token, str) else sep_token
        mask_token = (
            AddedToken(mask_token, normalized=False, special=True) if isinstance(mask_token, str) else mask_token
        )

        super().__init__(
            vocab_file,
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
            sp_model_kwargs=sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            add_prefix_space=add_prefix_space,
            legacy=legacy,
            **kwargs,
        )

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # TODO: check what is used as bos/eos as base PT uses no special tokens
        # e.g. cls/sep are used instead in the instruct tokenizers
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output


__all__ = ["Ernie4_5Tokenizer"]
