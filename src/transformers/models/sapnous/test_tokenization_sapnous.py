# coding=utf-8
# Copyright 2025-present, the HuggingFace Inc. Team and AIRAS Inc. Team. All rights reserved.
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
import unittest
from transformers import AutoTokenizer
from .configuration_sapnous import SapnousT1Config
from .tokenization_sapnous import SapnousT1Tokenizer

class TestSapnousTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = SapnousT1Tokenizer(
            vocab_file="vocab.json",
            merges_file="merges.txt"
        )

    def test_tokenizer_from_pretrained(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "Sapnous-AI/Sapnous-VR-6B",
            trust_remote_code=True
        )
        self.assertIsInstance(tokenizer, SapnousT1Tokenizer)

    def test_save_load_pretrained(self):
        vocab = self.tokenizer.get_vocab()
        self.assertIsInstance(vocab, dict)
        self.assertGreater(len(vocab), 0)

    def test_tokenization(self):
        text = "Hello, world!"
        tokens = self.tokenizer.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_special_tokens(self):
        self.assertIsNotNone(self.tokenizer.unk_token)
        self.assertIsNotNone(self.tokenizer.bos_token)
        self.assertIsNotNone(self.tokenizer.eos_token)

if __name__ == '__main__':
    unittest.main()