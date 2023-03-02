# coding=utf-8
# Copyright 2023 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
import json
from typing import TYPE_CHECKING, List, Optional, Tuple

from tokenizers import pre_tokenizers
import sentencepiece as spm

from ... import PreTrainedTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging


if TYPE_CHECKING:
    from transformers.pipelines.conversational import Conversation


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
}


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
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
    ) -> None:
        self.sp_model = spm.SentencePieceProcessor(model_file=vocab_file)
        # TODO @thomasw21: Understand if I need to have <bos> and such since they are not part of the official LLaMa model
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            # TODO @thomasw21: Why the fuck is that `-1`?
            pad_token=self.sp_model.pad_id(),
        )

        self.vocab_file = vocab_file

        self.sp_model = spm.SentencePieceProcessor(vocab_file)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        return [self.bos_token_id] + self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str):
        """Converts a token (str) in an id using the vocab."""
        # TODO @thomasw21: This means that you can't get the tokens from <bos>/<eos>/<unk>
        # The issue is that you can tokenizer <
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)
