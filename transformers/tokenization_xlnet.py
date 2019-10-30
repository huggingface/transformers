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
# limitations under the License.
""" Tokenization classes for XLNet model."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import os
from shutil import copyfile

import unicodedata
import six

from .tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'spiece.model'}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
    'xlnet-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-spiece.model",
    'xlnet-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-spiece.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'xlnet-base-cased': None,
    'xlnet-large-cased': None,
}

SPIECE_UNDERLINE = u'▁'

# Segments (not really needed)
SEG_ID_A   = 0
SEG_ID_B   = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

class XLNetTokenizer(PreTrainedTokenizer):
    """
        SentencePiece based tokenizer. Peculiarities:

            - requires `SentencePiece <https://github.com/google/sentencepiece>`_
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file,
                 do_lower_case=False, remove_space=True, keep_accents=False,
                 bos_token="<s>", eos_token="</s>", unk_token="<unk>", sep_token="<sep>",
                 pad_token="<pad>", cls_token="<cls>", mask_token="<mask>",
                 additional_special_tokens=["<eop>", "<eod>"], **kwargs):
        super(XLNetTokenizer, self).__init__(bos_token=bos_token, eos_token=eos_token,
                                             unk_token=unk_token, sep_token=sep_token,
                                             pad_token=pad_token, cls_token=cls_token,
                                             mask_token=mask_token, additional_special_tokens=
                                             additional_special_tokens, **kwargs)

        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens

        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning("You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                           "pip install sentencepiece")

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning("You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                           "pip install sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = ' '.join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if six.PY2 and isinstance(outputs, str):
            outputs = outputs.decode('utf-8')

        if not self.keep_accents:
            outputs = unicodedata.normalize('NFKD', outputs)
            outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text, return_unicode=True, sample=False):
        """ Tokenize a string.
            return_unicode is used only for py2
        """
        text = self.preprocess_text(text)
        # note(zhiliny): in some systems, sentencepiece only accepts str for py2
        if six.PY2 and isinstance(text, unicode):
            text = text.encode('utf-8')

        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(
                    piece[:-1].replace(SPIECE_UNDERLINE, ''))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        # note(zhiliny): convert back to unicode for py2
        if six.PY2 and return_unicode:
            ret_pieces = []
            for piece in new_pieces:
                if isinstance(piece, str):
                    piece = piece.decode('utf-8')
                ret_pieces.append(piece)
            new_pieces = ret_pieces

        return new_pieces

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index, return_unicode=True):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        if six.PY2 and return_unicode and isinstance(token, str):
            token = token.decode('utf-8')
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        An XLNet sequence has the following format:
            single sequence: X <sep> <cls>
            pair of sequences: A <sep> B <sep> <cls>
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return token_ids_0 + sep + cls
        return token_ids_0 + sep + token_ids_1 + sep + cls

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
                raise ValueError("You should not supply a second sequence if the provided sequence of "
                                 "ids is already formated with special tokens for the model.")
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1, 1]
        return ([0] * len(token_ids_0)) + [1, 1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        An XLNet sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 2
        | first sequence    | second sequence     | CLS segment ID
        
        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        cls_segment_id = [2]

        if token_ids_1 is None:
            return len(token_ids_0 + sep + cls) * [0]
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id

    def save_vocabulary(self, save_directory):
        """ Save the sentencepiece vocabulary (copy original file) and special tokens file
            to a directory.
        """
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        out_vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES['vocab_file'])

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
