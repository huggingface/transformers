# coding=utf-8
# Copyright Vasudev Gupta and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for BigBird."""


import os
from shutil import copyfile
from typing import List, Optional, Tuple

import sentencepiece as spm

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "gpt2.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"google/bigbird-base": "https://huggingface.co/google/bigbird-base/resolve/main/gpt2.model"}
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/bigbird-base": 4096,
}


class BigBirdTokenizer(PreTrainedTokenizer):
    """
    Construct a BigBird tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The begin of sequence token.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="</s>",
        eos_token="<s>",
        pad_token="<pad>",
        sep_token="<::::>",  # TODO: confirm this
        **kwargs
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            **kwargs,
        )

        self.vocab_file = vocab_file

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text, sample=False):
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        return pieces

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = self.sp_model.decode_pieces(tokens)
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    # TODO: next 4 methods not there in bertgeneration
    # def build_inputs_with_special_tokens(
    #         self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # ) -> List[int]:
    #     """
    # Build model inputs from a sequence or a pair of sequence for sequence classification tasks    # by concatenating and adding special tokens.
    # A BigBird sequence has the following format:

    # - single sequence: ``<s> X </s>`` # - pair of sequences: ``<s> A </s></s> B </s>``

    # Args: # token_ids_0 (:obj:`List[int]`): # List of IDs to which the special tokens will be added. # token_ids_1
    # (:obj:`List[int]`, `optional`):
    # Optional second list of IDs for sequence pairs.

    # Returns: # :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special
    # tokens.
    #     if token_ids_1 is None:
    #         return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
    #     cls = [self.cls_token_id]
    #     sep = [self.sep_token_id]
    #     return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    # def get_special_tokens_mask(
    #         self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    # ) -> List[int]:
    # Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding #
    # special tokens using the tokenizer ``prepare_for_model`` method.

    # Args: 
    # token_ids_0 (:obj:`List[int]`): 
    # List of IDs. 
    # token_ids_1 (:obj:`List[int]`, `optional`): 
    # Optional
    # second list of IDs for sequence pairs. 
    # already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`): 
    # Whether or not the token list is already formatted with special tokens for the model.

    # Returns: # :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence
    # token.
    #     if already_has_special_tokens:
    #         if token_ids_1 is not None:
    #             raise ValueError(
    #                 "You should not supply a second sequence if the provided sequence of "
    #                 "ids is already formatted with special tokens for the model."
    #             )
    #         return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

    #     if token_ids_1 is None:
    #         return [1] + ([0] * len(token_ids_0)) + [1]
    #     return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    # def create_token_type_ids_from_sequences(
    #         self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # ) -> List[int]:
    #     """
    # Create a mask from the two sequences passed to be used in a sequence-pair classification task. 
    # BigBird does not
    # make use of token type ids, therefore a list of zeros is returned.

    # Args: # token_ids_0 (:obj:`List[int]`): # List of IDs. # token_ids_1 (:obj:`List[int]`, `optional`): # Optional
    # second list of IDs for sequence pairs.

    # Returns: # :obj:`List[int]`: List of zeros. #
    #     sep = [self.sep_token_id]
    #     cls = [self.cls_token_id]

    #     if token_ids_1 is None:
    #         return len(cls + token_ids_0 + sep) * [0]
    #     return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    # def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
    #     add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
    #     if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
    #         text = " " + text
    #     return (text, kwargs)
