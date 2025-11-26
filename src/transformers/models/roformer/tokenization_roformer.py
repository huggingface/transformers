# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for RoFormer."""

from tokenizers import NormalizedString, PreTokenizedString, Tokenizer, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


class JiebaPreTokenizer:
    def __init__(self, vocab) -> None:
        self.vocab = vocab
        self.normalizers = normalizers.BertNormalizer(
            clean_text=False,
            handle_chinese_chars=True,
            strip_accents=False,
            lowercase=False,
        )
        try:
            import rjieba
        except ImportError:
            raise ImportError(
                "You need to install rjieba to use RoFormerTokenizer. "
                "See https://pypi.org/project/rjieba/ for installation."
            )
        self.jieba = rjieba

    def jieba_split(self, i: int, normalized_string: NormalizedString) -> list[NormalizedString]:
        splits = []

        # this code slice normalized_string is too slow (6s) but test_alignment_methods can pass
        for token, start, end in self.jieba.tokenize(str(normalized_string), hmm=False):
            if token in self.vocab:
                splits.append(normalized_string[start:end])
            else:
                token_list = self.normalizers.normalize_str(token).split()
                for token in token_list:
                    if token:
                        end = start + len(token)
                        splits.append(normalized_string[start:end])
                        start = end

        # this code test_alignment_methods can't pass but fast (300ms)
        # for token in self.jieba.cut(str(normalized_string), False):
        #     if token in self.vocab:
        #         splits.append(NormalizedString(token))
        #     else:
        #         token_list = self.normalizers.normalize_str(token).split()
        #         for token in token_list:
        #             if token:
        #                 splits.append(NormalizedString(token))

        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.jieba_split)


class RoFormerTokenizer(TokenizersBackend):
    r"""
    Construct a "fast" RoFormer tokenizer (backed by HuggingFace's *tokenizers* library).

    [`RoFormerTokenizerFast`] is almost identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece. There are some difference between them when tokenizing Chinese.

    This tokenizer inherits from [`PythonBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Example:

    ```python
    >>> from transformers import RoFormerTokenizerFast

    >>> tokenizer = RoFormerTokenizerFast.from_pretrained("junnyu/roformer_chinese_base")
    >>> tokenizer.tokenize("今天天气非常好。")
    ['今', '天', '天', '气', '非常', '好', '。']
    ```"""

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        if vocab is None:
            vocab = {
                str(unk_token): 0,
                str(pad_token): 1,
                str(cls_token): 2,
                str(sep_token): 3,
                str(mask_token): 4,
            }
        elif unk_token not in vocab:
            vocab[str(unk_token)] = len(vocab)


        self._tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(unk_token)))
        self._tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(JiebaPreTokenizer(vocab))

        self.do_lower_case = do_lower_case
        self.strip_accents = strip_accents

        super().__init__(
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
        cls = str(cls_token)
        sep = str(sep_token)
        cls_token_id = self.convert_tokens_to_ids(cls_token)
        sep_token_id = self.convert_tokens_to_ids(sep_token)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     state["_tokenizer"].pre_tokenizer = BertPreTokenizer()
    #     return state

    # def __setstate__(self, d):
    #     self.__dict__ = d
    #     vocab = self.__dict__["_tokenizer"].get_vocab()
    #     self.__dict__["_tokenizer"].pre_tokenizer = PreTokenizer.custom(JiebaPreTokenizer(vocab))

    def save_pretrained(
        self,
        save_directory,
        legacy_format=None,
        filename_prefix=None,
        push_to_hub=False,
        **kwargs,
    ):
        pre_tokenizer = self._tokenizer.pre_tokenizer
        self._tokenizer.pre_tokenizer = None
        result = super().save_pretrained(save_directory, legacy_format, filename_prefix, push_to_hub, **kwargs)
        self._tokenizer.pre_tokenizer = pre_tokenizer
        return result


__all__ = ["RoFormerTokenizer"]
