# coding=utf-8
# Copyright 2020 HuggingFace Inc. team.
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


import os
import unittest
from functools import lru_cache

from transformers import FunnelTokenizer, FunnelTokenizerFast
from transformers.models.funnel.tokenization_funnel import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin, use_cache_if_possible


@require_tokenizers
class FunnelTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "funnel-transformer/small"
    tokenizer_class = FunnelTokenizer
    rust_tokenizer_class = FunnelTokenizerFast
    test_rust_tokenizer = True
    space_between_special_tokens = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        vocab_tokens = [
            "<unk>",
            "<cls>",
            "<sep>",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
        ]
        cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(cls.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    @classmethod
    @use_cache_if_possible
    @lru_cache(maxsize=64)
    def get_tokenizer(cls, pretrained_name=None, **kwargs):
        pretrained_name = pretrained_name or cls.tmpdirname
        return FunnelTokenizer.from_pretrained(pretrained_name, **kwargs)

    @classmethod
    @use_cache_if_possible
    @lru_cache(maxsize=64)
    def get_rust_tokenizer(cls, pretrained_name=None, **kwargs):
        pretrained_name = pretrained_name or cls.tmpdirname
        return FunnelTokenizerFast.from_pretrained(pretrained_name, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "UNwant\u00e9d,running"
        output_text = "unwanted, running"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        tokens = tokenizer.tokenize("UNwant\u00e9d,running")
        self.assertListEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

    def test_token_type_ids(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            inputs = tokenizer("UNwant\u00e9d,running")
            sentence_len = len(inputs["input_ids"]) - 1
            self.assertListEqual(inputs["token_type_ids"], [2] + [0] * sentence_len)

            inputs = tokenizer("UNwant\u00e9d,running", "UNwant\u00e9d,running")
            self.assertListEqual(inputs["token_type_ids"], [2] + [0] * sentence_len + [1] * sentence_len)
