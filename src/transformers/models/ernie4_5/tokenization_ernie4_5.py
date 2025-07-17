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

from typing import Optional, Union

from ...tokenization_utils import AddedToken
from ...utils import logging
from ...utils.import_utils import requires
from ..llama import LlamaTokenizer


logger = logging.get_logger(__name__)


DEFAULT_CHAT_TEMPLATE = '{%- if not add_generation_prompt is defined -%}\n    {%- set add_generation_prompt = true -%}\n{%- endif -%}\n{%- if not cls_token is defined -%}\n    {%- set cls_token = "<|begin_of_sentence|>" -%}\n{%- endif -%}\n{%- if not sep_token is defined -%}\n    {%- set sep_token = "<|end_of_sentence|>" -%}\n{%- endif -%}\n{{- cls_token -}}\n{%- for message in messages -%}\n    {%- if message["role"] == "user" -%}\n        {{- "User: " + message["content"] + "\n" -}}\n    {%- elif message["role"] == "assistant" -%}\n        {{- "Assistant: " + message["content"] + sep_token -}}\n    {%- elif message["role"] == "system" -%}\n        {{- message["content"] + "\n" -}}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{- "Assistant: " -}}\n{%- endif -%}'


@requires(backends=("sentencepiece",))
class Ernie4_5Tokenizer(LlamaTokenizer):
    # NOTE: Wrapper around Llama to enable different default values and warn
    # if unintended behavior might be encounters, e.g. `add_prefix_space`.
    #
    # The `Ernie4_5Tokenizer` has to overwrite the `Llama` counterparts due
    # to the way the original tokenizer treats spacing as well as how special
    # tokens are treated. As many tokens have been added through other legacy
    # methods, we need to circumvent those behaviors here. See the docstrings
    # on the overwritten methods with example strings that would fail otherwise.

    legacy = True
    padding_side = "left"
    add_prefix_space = False

    def __init__(
        self,
        vocab_file,
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

        def convert_to_added_token(token):
            return AddedToken(token, normalized=False, special=True) if isinstance(token, str) else token

        cls_token = convert_to_added_token(cls_token)
        sep_token = convert_to_added_token(sep_token)
        mask_token = convert_to_added_token(mask_token)

        super().__init__(
            vocab_file,
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

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) in a single string.

        Overwritten to fix wrong behavior around adding spaces during decoding, e.g.
        when decoding " <s> Hello<s> how".
        """
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def _decode(
        self,
        token_ids: Union[int, list[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        """
        Overwritten as we want to handle the legacy-added tokens as normal tokens instead.
        The easiest way to see the issue is by decoding sequences of numbers or additional special
        tokens (e.g. "15 663" or "<|LOC_BEGIN|>test<|LOC_END|>").
        """
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        # If given is a single id, prevents splitting the string in upcoming loop
        if isinstance(filtered_tokens, str):
            filtered_tokens = [filtered_tokens]

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_tokens:
                continue
            current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text


__all__ = ["Ernie4_5Tokenizer"]
