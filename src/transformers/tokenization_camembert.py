# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
# limitations under the License
""" Tokenization classes for Camembert model."""


import logging
import os
from shutil import copyfile

import sentencepiece as spm

from transformers.tokenization_utils import PreTrainedTokenizer

from .tokenization_xlnet import SPIECE_UNDERLINE


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "camembert-base": "https://s3.amazonaws.com/models.huggingface.co/bert/camembert-base-sentencepiece.bpe.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "camembert-base": None,
}

SHARED_MODEL_IDENTIFIERS = [
    # Load with
    # `tokenizer = AutoTokenizer.from_pretrained("username/pretrained_model")`
    "Musixmatch/umberto-commoncrawl-cased-v1",
    "Musixmatch/umberto-wikipedia-uncased-v1",
]


class CamembertTokenizer(PreTrainedTokenizer):
    """
        Adapted from RobertaTokenizer and XLNetTokenizer
        SentencePiece based tokenizer. Peculiarities:

            - requires `SentencePiece <https://github.com/google/sentencepiece>`_
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["<s>NOTUSED", "</s>NOTUSED"],
        **kwargs
    ):
        super().__init__(
            max_len=512,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 4  # take into account special tokens
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file
        # HACK: These tokens were added by fairseq but don't seem to be actually used when duplicated in the actual
        # sentencepiece vocabulary (this is the case for <s> and </s>
        self.fairseq_tokens_to_ids = {"<s>NOTUSED": 0, "<pad>": 1, "</s>NOTUSED": 2, "<unk>": 3}
        self.fairseq_offset = len(self.fairseq_tokens_to_ids)
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + len(self.fairseq_tokens_to_ids)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:
            single sequence: <s> X </s>
            pair of sequences: <s> A </s></s> B </s>
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

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

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A RoBERTa sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep) * [0] + len(token_ids_1 + sep) * [1]

    @property
    def vocab_size(self):
        return len(self.fairseq_tokens_to_ids) + len(self.sp_model)

    def _tokenize(self, text):
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        elif self.sp_model.PieceToId(token) == 0:
            # Convert sentence piece unk token to fairseq unk token index
            return self.unk_token_id
        return self.fairseq_offset + self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use AlbertTokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece"
            )
            raise
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(self, save_directory):
        """ Save the sentencepiece vocabulary (copy original file) and special tokens file
            to a directory.
        """
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        out_vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
