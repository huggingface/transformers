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
"""Tokenization utils for RoFormer."""

from tokenizers import NormalizedString, PreTokenizedString, normalizers


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
