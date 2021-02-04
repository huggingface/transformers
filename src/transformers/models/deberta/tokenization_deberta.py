# coding=utf-8
# Copyright 2020 Microsoft and the HuggingFace Inc. team.
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

import os
from typing import Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer
from .gpt2_tokenizer import GPT2Tokenizer
from .spm_tokenizer import SPMTokenizer


PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/bpe_encoder.bin",
        "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/bpe_encoder.bin",
        "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/bpe_encoder.bin",
        "microsoft/deberta-xlarge-v2": "https://huggingface.co/microsoft/deberta-xlarge-v2/resolve/main/spm.model",
        "microsoft/deberta-xxlarge-v2": "https://huggingface.co/microsoft/deberta-xxlarge-v2/resolve/main/spm.model",
        "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/bpe_encoder.bin",
        "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/bpe_encoder.bin",
        "microsoft/deberta-xlarge-mnli": "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/bpe_encoder.bin",
        "microsoft/deberta-xlarge-v2-mnli": "https://huggingface.co/microsoft/deberta-xlarge-v2-mnli/resolve/main/spm.model",
        "microsoft/deberta-xxlarge-v2-mnli": "https://huggingface.co/microsoft/deberta-xxlarge-v2-mnli/resolve/main/spm.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-base": 512,
    "microsoft/deberta-large": 512,
    "microsoft/deberta-xlarge": 512,
    "microsoft/deberta-xlarge-v2": 512,
    "microsoft/deberta-xxlarge-v2": 512,
    "microsoft/deberta-base-mnli": 512,
    "microsoft/deberta-large-mnli": 512,
    "microsoft/deberta-xlarge-mnli": 512,
    "microsoft/deberta-xlarge-v2-mnli": 512,
    "microsoft/deberta-xxlarge-v2-mnli": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-base": {"do_lower_case": False, "vocab_type": "gpt2"},
    "microsoft/deberta-large": {"do_lower_case": False, "vocab_type": "gpt2"},
    "microsoft/deberta-xlarge": {"do_lower_case": False, "vocab_type": "gpt2"},
    "microsoft/deberta-xlarge-v2": {"do_lower_case": False, "vocab_type": "spm"},
    "microsoft/deberta-xxlarge-v2": {"do_lower_case": False, "vocab_type": "spm"},
    "microsoft/deberta-base-mnli": {"do_lower_case": False, "vocab_type": "gpt2"},
    "microsoft/deberta-large-mnli": {"do_lower_case": False, "vocab_type": "gpt2"},
    "microsoft/deberta-xlarge-mnli": {"do_lower_case": False, "vocab_type": "gpt2"},
    "microsoft/deberta-xlarge-v2-mnli": {"do_lower_case": False, "vocab_type": "spm"},
    "microsoft/deberta-xxlarge-v2-mnli": {"do_lower_case": False, "vocab_type": "spm"},
}

__all__ = ["DebertaTokenizer"]

VOCAB_FILES_NAMES = {"vocab_file": "bpe_encoder.bin"}


class DebertaTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a DeBERTa tokenizer, which runs end-to-end tokenization: punctuation splitting + wordpiece

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        vocab_type="gpt2",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__(
            do_lower_case=do_lower_case,
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
        if vocab_type.lower() == "gpt2":
            self._tokenizer = GPT2Tokenizer(vocab_file, **kwargs)
        else:
            self._tokenizer = SPMTokenizer(vocab_file, **kwargs)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab(self):
        return self._tokenizer.vocab

    def get_vocab(self):
        vocab = self.vocab.copy()
        vocab.update(self.get_added_vocab())
        return vocab

    def _tokenize(self, text):
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        if self.do_lower_case:
            text = text.lower()
        return self._tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self._tokenizer.sym(index) if index < self.vocab_size else self.unk_token

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        return self._tokenizer.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTa sequence has the following format:

        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
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
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(
                    lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0,
                )
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa
        sequence pair mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return self._tokenizer.save_pretrained(save_directory, filename_prefix=filename_prefix)
