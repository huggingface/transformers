# coding=utf-8
# Copyright 2018 HuggingFace Inc. team.
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

from transformers import RemBertTokenizer, RemBertTokenizerFast
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, slow
from transformers.utils import is_torch_available

from ...test_tokenization_common import TokenizerTesterMixin


#SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")
#SAMPLE_BPE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_bpe.model")

FRAMEWORK = "pt" if is_torch_available() else "tf"


@require_sentencepiece
@require_tokenizers
class RemBertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    
    tokenizer_class = RemBertTokenizer
    rust_tokenizer_class = RemBertTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        tokenizer = RemBertTokenizer.from_pretrained("google/rembert")
        tokenizer.save_pretrained(self.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "[PAD]"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "[PAD]")
        self.assertEqual(vocab_keys[1], "</s>")
        self.assertEqual(vocab_keys[-1], "ઞ")
        self.assertEqual(len(vocab_keys), 1_000)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_000)

    def test_rust_and_python_bpe_tokenizers(self):
        tokenizer = RemBertTokenizer.from_pretrained("google/rembert")
        tokenizer.save_pretrained(self.tmpdirname)
        rust_tokenizer = RemBertTokenizerFast.from_pretrained(self.tmpdirname)

        sequence = "Hi I am new at huggingface."

        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        # <unk> tokens are not the same for `rust` than for `slow`.
        # Because spm gives back raw token instead of `unk` in EncodeAsPieces
        # tokens = tokenizer.tokenize(sequence)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "I was born in 92000, and this is falsé."

        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

    """@slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[5, 54, 7196, 297, 30, 23, 776, 18, 11, 3215, 3705, 8252, 22, 3164, 1181, 2116, 29, 16, 813, 25, 791, 3314, 20, 3446, 38, 27575, 120, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [5, 468, 17, 11, 9088, 20, 1517, 8, 22804, 18818, 10, 38, 629, 607, 607, 142, 19, 7196, 867, 56, 10326, 24, 2267, 20, 416, 5072, 15612, 233, 734, 7, 2399, 27, 16, 3015, 1649, 7, 24, 20, 4338, 2399, 27, 13, 3400, 14, 13, 6189, 8, 930, 9, 6]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # noqa: E501
        # fmt: on

        sequences = [
            "Transformer is a deep learning model introduced in 2017,",
            "used mainly in the field of automatic language processing (NLP).",
            "Like recurrent neural networks (RNNs), transformers are designed ",
            "to handle sequential data, such as natural language, for tasks ",
            "such as text translation and synthesis.",
        ]

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="google/rembert",
            revision="3a0641d9a1aeb7e848a74299e7e4c4bca216b4cf",
            sequences=sequences,
        )"""