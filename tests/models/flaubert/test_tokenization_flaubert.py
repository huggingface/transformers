# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the FlauBERT tokenizer."""

import json
import os
import unittest

from transformers import FlaubertTokenizer
from transformers.models.flaubert.tokenization_flaubert import VOCAB_FILES_NAMES
from transformers.testing_utils import slow

from ...test_tokenization_common import TokenizerTesterMixin


class FlaubertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "flaubert/flaubert_base_cased"
    tokenizer_class = FlaubertTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n", "w</w>", "r</w>", "t</w>", "i</w>", "lo", "low", "ne", "new", "er</w>", "low</w>", "lowest</w>", "new</w>", "newer</w>", "wider</w>", "<unk>"]  # fmt: skip

        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["n e 300", "ne w 301", "e r</w> 302", ""]

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    # Copied from transformers.tests.models.xlm.test_tokenization_xlm.XLMTokenizationTest.test_full_tokenizer
    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()
        text = "lower newer"
        bpe_tokens = ["l", "o", "w", "er</w>", "new", "er</w>"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [0, 1, 2, 18, 17, 18, 24]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    @slow
    # Copied from transformers.tests.models.xlm.test_tokenization_xlm.XLMTokenizationTest.test_sequence_builders
    def test_sequence_builders(self):
        tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)
        print(encoded_sentence)
        print(encoded_sentence)

        assert encoded_sentence == [0] + text + [1]
        assert encoded_pair == [0] + text + [1] + text_2 + [1]
