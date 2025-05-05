# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Dia."""

from typing import List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


class DiaTokenizer(PreTrainedTokenizer):
    """
    Construct a "fast" Dia tokenizer (backed by HuggingFace's *tokenizers* library).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        max_length (`int`, *optional*, defaults to `1024`):
            The maximum length of the sequences when encoding. Sequences longer than this will be truncated.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        pad_token: Optional[str] = "<pad>",
        unk_token: Optional[str] = "<pad>",
        max_length: Optional[int] = 1024,
        **kwargs,
    ):
        self._utf_vocab_size = 2**8  # utf is 8 bits
        self._added_tokens_decoder = {0: pad_token, 1: "[S1]", 2: "[S2]"}
        self.offset = len(self._added_tokens_decoder)
        super().__init__(
            unk_token=pad_token,
            pad_token=pad_token,
            max_length=max_length,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self._utf_vocab_size

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size + self.offset)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""

        if len(token) != 1:
            token_id = None
        else:
            token_id = ord(token) + self.offset

        return token_id

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        bstring = b""
        for token in tokens:
            if token in self.added_tokens_decoder:
                tok_string = self.added_tokens_decoder[token].encode("utf-8")
            elif token in self.added_tokens_encoder:
                tok_string = token.encode("utf-8")
            else:
                tok_string = bytes([ord(token)])
            bstring += tok_string
        string = bstring.decode("utf-8", errors="ignore")
        return string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1]
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones


__all__ = ["DiaTokenizer"]
