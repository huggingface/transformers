# coding=utf-8
# Copyright 2023 HuggingFace Inc. team.
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
import shutil
import unittest

from transformers.models.scibart.tokenization_scibart import SciBartTokenizer
from transformers.testing_utils import get_tests_dir

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_bpe.model")


class SciBartTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = SciBartTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        vocab = ["▁This", "▁is", "▁a", "▁t", "est"]
        dict(zip(vocab, range(len(vocab))))
        self.special_tokens_map = {"unk_token": "<unk>"}

        tokenizer = SciBartTokenizer(SAMPLE_VOCAB, **self.special_tokens_map)
        tokenizer.save_pretrained(os.path.join(self.tmpdirname, "scibart_tokenizer"))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return SciBartTokenizer.from_pretrained(os.path.join(self.tmpdirname, "scibart_tokenizer"), **kwargs)

    def test_full_tokenizer(self):
        tokenizer = SciBartTokenizer.from_pretrained("uclanlp/scibart-base")

        text = "This paper proposes an <mask> for keyphrase generation."
        bpe_tokens = [
            "▁This",
            "▁paper",
            "▁proposes",
            "▁an",
            "<mask>",
            "▁for",
            "▁key",
            "ph",
            "rase",
            "▁generation",
            ".",
        ]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [330, 521, 5703, 91, 30001, 72, 1840, 190, 15681, 2740, 29912, 0]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)
