# coding=utf-8
# Copyright 2018 T5 Authors and HuggingFace Inc. team.
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
"""Tokenization classes for LLaMa."""
from typing import List, Optional, Union

import sentencepiece as spm

from ... import PreTrainedTokenizer, AddedToken
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


class LLaMaTokenizer(PreTrainedTokenizer):
    """
    Construct a "LLaMa", Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (`str`, *optional*, defaults to `"<bos>"`):
            The end of sequence token.
        eos_token (`str`, *optional*, defaults to `"<eos>"`):
            The beginning of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    padding_side: str = "left"

    def __init__(
        self, vocab_file: str, bos_token: str = "<bos>", eos_token: str = "<eos>", unk_token: str = "<unk>", **kwargs
    ) -> None:
        self.sp_model = spm.SentencePieceProcessor(model_file=vocab_file)

        # TODO @thomasw21: Understand if I need to have <bos> and such since they are not part of the official LLaMa model
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            # TODO @thomasw21: Why the fuck is that `-1`?
            # pad_token=self.sp_model.pad_id(),
            **kwargs,
        )

        self.vocab_file = vocab_file

        self.sp_model = spm.SentencePieceProcessor(vocab_file)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        # <bos>/<eos>/<unk> already have ids in LLaMa tokenizer
        new_tokens = [tok for tok in new_tokens if tok not in [self.bos_token, self.eos_token, self.unk_token]]
        return super()._add_tokens(new_tokens=new_tokens, special_tokens=special_tokens)


    def bos_token_id(self) -> Optional[int]:
        result = self.sp_model.bos_id()
        if result >= 0:
            return result
        else:
            return None
    def eos_token_id(self) -> Optional[int]:
        result = self.sp_model.eos_id()
        if result >= 0:
            return result
        else:
            return None
    def unk_token_id(self) -> Optional[int]:
        result = self.sp_model.unk_id()
        if result >= 0:
            return result
        else:
            return None
    def pad_token_id(self) -> Optional[int]:
        result = self.sp_model.pad_id()
        if result >= 0:
            return result
        else:
            return None

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `<bos> X`
        - pair of sequences: `<bos> A B`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        result = [self.sp_model.bos_id()] + token_ids_0
        if token_ids_1 is not None:
            result += token_ids_1
        return result

    def convert_tokens_to_string(self, tokens: List[str]):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()
