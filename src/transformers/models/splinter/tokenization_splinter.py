# coding=utf-8
# Copyright 2021 Tel AViv University, AllenAI and The HuggingFace Inc. team. All rights reserved.
# All rights reserved.
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
"""Tokenization classes for Splinter."""

import collections
from typing import Optional

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class SplinterTokenizer(TokenizersBackend):
    r"""
    Construct a Splinter tokenizer (backed by HuggingFace's tokenizers library). Based on WordPiece.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to a vocabulary file.
        tokenizer_file (`str`, *optional*):
            Path to a tokenizers JSON file containing the serialization of a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values.
        question_token (`str`, *optional*, defaults to `"[QUESTION]"`):
            The token used for constructing question representations.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase`.
        vocab (`dict`, *optional*):
            Custom vocabulary dictionary. If not provided, a minimal vocabulary is created.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    def __init__(
        self,
        do_lower_case: bool = True,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        question_token: str = "[QUESTION]",
        tokenize_chinese_chars: bool = True,
        strip_accents: Optional[bool] = None,
        vocab: Optional[dict] = None,
        **kwargs,
    ):
        if vocab is not None:
            self._vocab = (
                {token: idx for idx, (token, _score) in enumerate(vocab)} if isinstance(vocab, list) else vocab
            )
        else:
            self._vocab = {
                str(pad_token): 0,
                str(unk_token): 1,
                str(cls_token): 2,
                str(sep_token): 3,
                str(mask_token): 4,
                str(question_token): 5,
                ".": 6,
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
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            question_token=question_token,
            do_lower_case=do_lower_case,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        if hasattr(self, "_tokenizer") and self._tokenizer.normalizer is not None:
            import json

            pre_tok_state = json.loads(self._tokenizer.normalizer.__getstate__())
            if (
                pre_tok_state.get("lowercase", do_lower_case) != do_lower_case
                or pre_tok_state.get("strip_accents", strip_accents) != strip_accents
                or pre_tok_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
            ):
                pre_tok_class = getattr(normalizers, pre_tok_state.pop("type"))
                pre_tok_state["lowercase"] = do_lower_case
                pre_tok_state["strip_accents"] = strip_accents
                pre_tok_state["handle_chinese_chars"] = tokenize_chinese_chars
                self._tokenizer.normalizer = pre_tok_class(**pre_tok_state)

        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.question_token = question_token
        if self.question_token not in self.all_special_tokens:
            self.add_tokens([self.question_token], special_tokens=True)
        self.update_post_processor()

    @property
    def question_token_id(self):
        return self.convert_tokens_to_ids(self.question_token)

    def update_post_processor(self):
        cls = self.cls_token
        sep = self.sep_token
        question = self.question_token
        dot = "."
        cls_token_id = self.cls_token_id
        sep_token_id = self.sep_token_id
        question_token_id = self.question_token_id
        dot_token_id = self.convert_tokens_to_ids(".")

        if cls is None or sep is None:
            return

        if self.padding_side == "right":
            pair = f"{cls}:0 $A:0 {question} {dot} {sep}:0 $B:1 {sep}:1"
        else:
            pair = f"{cls}:0 $A:0 {sep}:0 $B:1 {question} {dot} {sep}:1"

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=pair,
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
                (question, question_token_id),
                (dot, dot_token_id),
            ],
        )


__all__ = ["SplinterTokenizer"]
