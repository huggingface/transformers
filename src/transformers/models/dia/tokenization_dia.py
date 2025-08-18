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
"""Tokenization class for Dia."""

from typing import Optional

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


class DiaTokenizer(PreTrainedTokenizer):
    """
    Construct a Dia tokenizer. Dia simply uses raw bytes utf-8 encoding except for special tokens `[S1]` and `[S2]`.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        unk_token (`str`, *optional*, defaults to `"<pad>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        max_length (`int`, *optional*, defaults to 1024):
            The maximum length of the sequences when encoding. Sequences longer than this will be truncated.
        offset (`int`, *optional*, defaults to 0):
            The offset of the tokenizer.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        pad_token: Optional[str] = "<pad>",
        unk_token: Optional[str] = "<pad>",
        max_length: Optional[int] = 1024,
        offset: int = 0,
        **kwargs,
    ):
        # We have no eos/bos tokens but allow padding -- no l/r strip as we treat them as tokens as well
        pad_token = AddedToken(pad_token) if isinstance(pad_token, str) else pad_token
        unk_token = AddedToken(unk_token) if isinstance(unk_token, str) else unk_token

        self._utf_vocab_size = 2**8  # utf is 8 bits
        self._added_tokens_decoder = {0: pad_token, 1: AddedToken("[S1]"), 2: AddedToken("[S2]")}
        self.offset = offset
        super().__init__(
            unk_token=unk_token,
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

    def _tokenize(self, text: str) -> list[str]:
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

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = chr(index - self.offset)
        return token

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        bstring = b""
        for token in tokens:
            if token in self.added_tokens_decoder:
                added_token_obj = self.added_tokens_decoder[token]
                tok_string = str(added_token_obj).encode("utf-8")
            elif token in self.added_tokens_encoder:
                tok_string = token.encode("utf-8")
            else:
                tok_string = token.encode("utf-8")  # Assume general string token
            bstring += tok_string
        string = bstring.decode("utf-8", errors="ignore")
        return string

    # No vocab file
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        return ()


__all__ = ["DiaTokenizer"]
