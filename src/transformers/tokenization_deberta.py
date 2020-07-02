# coding=utf-8
# Copyright 2020 Microsoft.
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
""" Tokenization class for model DeBERTa."""

import logging
import os

from DeBERTa.deberta import GPT2Tokenizer

from .tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)

####################################################
# Mapping from the keyword arguments names of Tokenizer `__init__`
# to file names for serializing Tokenizer instances
####################################################
VOCAB_FILES_NAMES = {"vocab_file": "bpe_encoder.bin"}

####################################################
# Mapping from the keyword arguments names of Tokenizer `__init__`
# to pretrained vocabulary URL for all the model shortcut names.
####################################################
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/microsoft/deberta-base/bpe_encoder.bin",
        "microsoft/deberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/microsoft/deberta-large/bpe_encoder.bin",
    }
}

####################################################
# Mapping from model shortcut names to max length of inputs
####################################################
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-base": 512,
    "microsoft/deberta-large": 512,
}

####################################################
# Mapping from model shortcut names to a dictionary of additional
# keyword arguments for Tokenizer `__init__`.
# To be used for checkpoint specific configurations.
####################################################
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-base": {"do_lower_case": False},
    "microsoft/deberta-large": {"do_lower_case": False},
}

__all__ = ["DeBERTaTokenizer"]


class DeBERTaTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a XxxTokenizer.
    :class:`~transformers.DeBERTaTokenizer` runs end-to-end tokenization: punctuation splitting + wordpiece

    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_basic_tokenize=True
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        """Constructs a XxxTokenizer.

        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file
            **do_lower_case**: (`optional`) boolean (default False)
                Whether to lower case the input
                Only has an effect when do_basic_tokenize=True
        """
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = XxxTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.do_lower_case = do_lower_case
        self.gpt2_tokenizer = GPT2Tokenizer(vocab_file)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab(self):
        return self.gpt2_tokenizer.vocab

    def get_vocab(self):
        vocab = self.vocab.copy()
        vocab.update(self.get_added_vocab())
        return vocab

    def _tokenize(self, text):
        """ Take as input a string and return a list of strings (tokens) for words/sub-words
        """
        if self.do_lower_case:
            text = text.lower()
        return self.gpt2_tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.gpt2_tokenizer.sym(index) if index < self.vocab_size else self.unk_token

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        return self.gpt2_tokenizer.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:

        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def prepare_for_tokenization(self, text, is_pretokenized=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        if is_pretokenized or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        self.gpt2_tokenizer.save_pretrained(vocab_file)
        return (vocab_file,)
