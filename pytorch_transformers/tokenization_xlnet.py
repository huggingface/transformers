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

import json
import logging
import os
import sys
from shutil import copyfile
from io import open

import unicodedata
import six

from .file_utils import cached_path
from .tokenization_utils import PreTrainedTokenizer, clean_up_tokenization

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'spiece.model'}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
    'xlnet-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-spiece.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'xlnet-large-cased': 512,
}

VOCAB_NAME = 'spiece.model'

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
            - requires SentencePiece: https://github.com/google/sentencepiece
    """
    # Tokens
    special_symbols = {
        "<unk>"  : 0,
        "<s>"    : 1,
        "</s>"   : 2,
        "<cls>"  : 3,
        "<sep>"  : 4,
        "<pad>"  : 5,
        "<mask>" : 6,
        "<eod>"  : 7,
        "<eop>"  : 8,
    }
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, max_len=None,
                 do_lower_case=False, remove_space=True, keep_accents=False):
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning("You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                           "pip install sentencepiece")

        self.max_len = max_len if max_len is not None else int(1e12)
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def UNK_TOKEN(self):
        return "<unk>"

    @property
    def SEP_TOKEN(self):
        return "<sep>"

    @property
    def PAD_TOKEN(self):
        return "<pad>"

    @property
    def CLS_TOKEN(self):
        return "<cls>"

    @property
    def MASK_TOKEN(self):
        return "<mask>"

    @property
    def UNK_ID(self):
        return self.special_symbols["<unk>"]

    @property
    def SEP_ID(self):
        return self.special_symbols["<sep>"]

    @property
    def PAD_ID(self):
        return self.special_symbols["<pad>"]

    @property
    def CLS_ID(self):
        return self.special_symbols["<cls>"]

    @property
    def MASK_ID(self):
        return self.special_symbols["<mask>"]

    def __len__(self):
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

    def tokenize(self, text, return_unicode=True, sample=False):
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

    def convert_tokens_to_ids(self, tokens, sample=False):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        if isinstance(tokens, str) or (sys.version_info[0] == 2 and isinstance(tokens, unicode)):
            return self.sp_model.PieceToId(tokens)
        for token in tokens:
            ids.append(self.sp_model.PieceToId(token))
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this XLNet model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids, return_unicode=True):
        """Converts a sequence of ids in tokens."""
        tokens = []
        for i in ids:
            tokens.append(self.sp_model.IdToPiece(i))

        if six.PY2 and return_unicode:
            ret_pieces = []
            for piece in tokens:
                if isinstance(piece, str):
                    piece = piece.decode('utf-8')
                ret_pieces.append(piece)
            tokens = ret_pieces
        return tokens

    def encode(self, text, sample=False):
        return self.convert_tokens_to_ids(self.tokenize(text, sample=sample))

    def decode(self, ids, clean_up_tokenization_spaces=True):
        """Converts a sequence of ids in a string."""
        tokens = self.convert_ids_to_tokens(ids)
        out_string = ''.join(tokens)
        if clean_up_tokenization_spaces:
            out_string = out_string.strip().replace('<unk>', '')
            out_string = clean_up_tokenization(out_string)
        return out_string

    def save_vocabulary(self, vocab_path):
        """ Save the sentencepiece vocabulary (copy original file) and special tokens file
            to a directory.
        """
        if not os.path.isdir(vocab_path):
            logger.error("Vocabulary path ({}) should be a directory".format(vocab_path))
            return
        out_vocab_file = os.path.join(vocab_path, VOCAB_NAME)

        copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
