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
"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from .tokenization_bert import BertTokenizer

logger = logging.getLogger(__name__)


class RBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None,
                 unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]", tokenize_chinese_chars=True, **kwargs):
        super(RBertTokenizer, self).__init__(vocab_file,
                                             do_lower_case=do_lower_case,
                                             do_basic_tokenize=do_basic_tokenize,
                                             never_split=never_split,
                                             unk_token=unk_token,
                                             sep_token=sep_token,
                                             pad_token=pad_token,
                                             cls_token=cls_token,
                                             mask_token=mask_token,
                                             tokenize_chinese_chars=tokenize_chinese_chars,
                                             **kwargs)
        self.ent1_sep_token = "$"
        self.ent2_sep_token = "#"
        self.ent1_sep_token_id = self.encode(self.ent1_sep_token)[0]
        self.ent2_sep_token_id = self.encode(self.ent2_sep_token)[0]

    def encode_with_relationship(self, text, e1_offset_tup, e2_offset_tup,
                                 text_pair=None, add_special_tokens=False):

        if e1_offset_tup and e2_offset_tup:
            text = self._clean_special_chars(text)
            text = self._insert_relationship_special_chars(text, e1_offset_tup, e2_offset_tup)

        return self.encode(text=text, text_pair=text_pair, add_special_tokens=add_special_tokens)

    def _clean_special_chars(self, text: str):
        return text.replace(self.ent1_sep_token, '|').replace(self.ent2_sep_token, '|')

    def _insert_relationship_special_chars(self, text, e1_offset_tup, e2_offset_tup):
        reverse_ordered_tups = sorted([e1_offset_tup, e2_offset_tup], reverse=True, key=lambda x: x[1])
        last_token_to_process = True
        for start, end in reverse_ordered_tups:
            before = text[:start]
            during = text[start:end]
            after = text[end:]
            if last_token_to_process:
                text = before + self.ent2_sep_token + " " + during + self.ent2_sep_token + " " + after
            else:
                text = before + self.ent1_sep_token + " " + during + self.ent1_sep_token + " " + after
            last_token_to_process = False
        return text
