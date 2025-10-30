# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for Bert."""

import collections
from typing import Optional

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import WordPiece

from ...tokenization_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class BertTokenizer(TokenizersBackend):
    r"""
    Construct a BERT tokenizer (backed by HuggingFace's tokenizers library). Based on WordPiece.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            File containing the vocabulary.
        tokenizer_file (`str`, *optional*):
            Path to a tokenizers JSON file containing the serialization of a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        vocab (`dict`, *optional*):
            Custom vocabulary dictionary. If not provided, vocabulary is loaded from vocab_file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    slow_tokenizer_class = None

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        do_lower_case: bool = False,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        tokenize_chinese_chars: bool = True,
        strip_accents: Optional[bool] = None,
        vocab: Optional[dict] = None,
        **kwargs,
    ):
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

        if vocab is not None:
            self._vocab = {token: idx for idx, (token, _score) in enumerate(vocab)} if isinstance(vocab, list) else vocab
        else:
            self._vocab = {
                str(pad_token): 0,
                str(unk_token): 1,
                str(cls_token): 2,
                str(sep_token): 3,
                str(mask_token): 4,
            }

        self._tokenizer = Tokenizer(WordPiece(self._vocab, unk_token=str(unk_token)))

        self._tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self._tokenizer.decoder = decoders.WordPiece(prefix="##")

        tokenizer_object = self._tokenizer

        super().__init__(
            tokenizer_object=tokenizer_object,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )


__all__ = ["BertTokenizer"]
