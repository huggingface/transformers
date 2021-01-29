# coding=utf-8
# Copyright 2018 Microsoft, the Hugging Face Team.
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


import re
import unittest
from typing import Tuple

from transformers.models.deberta.tokenization_deberta import DebertaTokenizer
from transformers.testing_utils import require_torch

from .test_tokenization_common import TokenizerTesterMixin


@require_torch
class DebertaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = DebertaTokenizer

    def setUp(self):
        super().setUp()

    def get_tokenizer(self, name="microsoft/deberta-base", **kwargs):
        return DebertaTokenizer.from_pretrained(name, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20) -> Tuple[str, list]:
        toks = [
            (i, tokenizer.decode([i], clean_up_tokenization_spaces=False))
            for i in range(5, min(len(tokenizer), 50260))
        ]
        toks = list(filter(lambda t: re.match(r"^[ a-zA-Z]+$", t[1]), toks))
        toks = list(filter(lambda t: [t[0]] == tokenizer.encode(t[1], add_special_tokens=False), toks))
        if max_length is not None and len(toks) > max_length:
            toks = toks[:max_length]
        # toks_str = [t[1] for t in toks]
        toks_ids = [t[0] for t in toks]

        # Ensure consistency
        output_txt = tokenizer.decode(toks_ids, clean_up_tokenization_spaces=False)
        if " " not in output_txt and len(toks_ids) > 1:
            output_txt = (
                tokenizer.decode([toks_ids[0]], clean_up_tokenization_spaces=False)
                + " "
                + tokenizer.decode(toks_ids[1:], clean_up_tokenization_spaces=False)
            )
        if with_prefix_space and not output_txt.startswith(" "):
            output_txt = " " + output_txt
        output_ids = tokenizer.encode(output_txt, add_special_tokens=False)
        return output_txt, output_ids

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer("microsoft/deberta-base")
        input_str = "UNwant\u00E9d,running"
        tokens = tokenizer.tokenize(input_str)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        self.assertEqual(tokenizer.decode(token_ids), input_str)
