# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

from tests.test_tokenization_common import TokenizerTesterMixin
from transformers import SplinterTokenizerFast, is_torch_available, is_tf_available
from transformers.models.splinter import SplinterTokenizer
from transformers.testing_utils import get_tests_dir, slow

SAMPLE_VOCAB = get_tests_dir("fixtures/vocab_splinter.txt")

if is_torch_available():
    FRAMEWORK = "pt"
elif is_tf_available():
    FRAMEWORK = "tf"
else:
    FRAMEWORK = "jax"


class SplinterTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = SplinterTokenizer
    rust_tokenizer_class = SplinterTokenizerFast
    space_between_special_tokens = False
    test_rust_tokenizer = False
    test_sentencepiece_ignore_case = False
    pre_trained_model_path = "tau/splinter-base"

    def setUp(self):
        super().setUp()
        tokenizer = SplinterTokenizer(SAMPLE_VOCAB)
        tokenizer.add_tokens("this")
        tokenizer.add_tokens("is")
        tokenizer.add_tokens("a")
        tokenizer.add_tokens("test")
        tokenizer.add_tokens("thou")
        tokenizer.add_tokens("shall")
        tokenizer.add_tokens("not")
        tokenizer.add_tokens("determine")
        tokenizer.add_tokens("rigor")
        tokenizer.add_tokens("truly")
        # tokenizer.add_tokens(tokenizer.question_token, special_tokens=True)
        # tokenizer.add_tokens(".")
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs) -> SplinterTokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs) -> SplinterTokenizerFast:
        return self.rust_tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())
        self.assertEqual(vocab_keys[0], "[PAD]")
        self.assertEqual(vocab_keys[1], "[SEP]")
        self.assertEqual(vocab_keys[2], "[MASK]")

    def test_convert_token_and_id(self):
        token = "[PAD]"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_question_token_id(self):
        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.question_token_id, tokenizer.convert_tokens_to_ids(tokenizer.question_token))

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()
        test_str = "This is a test"

        unk_token = tokenizer.unk_token
        unk_token_id = tokenizer._convert_token_to_id_with_added_voc(unk_token)

        expected_tokens = test_str.lower().split()
        tokenizer.add_tokens(expected_tokens)
        tokens = tokenizer.tokenize(test_str)
        self.assertListEqual(tokens, expected_tokens)

        # test with out of vocabulary string
        tokens = tokenizer.tokenize(test_str + " oov")
        self.assertListEqual(tokens, expected_tokens + [unk_token])

        expected_token_ids = [13, 14, 15, 16, unk_token_id]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(token_ids, expected_token_ids)

        tokenizer = self.get_tokenizer(basic_tokenize=False)
        expected_token_ids = [13, 14, 15, 16, unk_token_id]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(token_ids, expected_token_ids)

    def test_rust_and_python_full_tokenizers(self):
        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "I need to test this rigor"
        tokens = tokenizer.tokenize(sequence, add_special_tokens=False)
        rust_tokens = rust_tokenizer.tokenize(sequence, add_special_tokens=False)
        self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

    def test_max_length(self):
        max_length = 20
        tokenizer = self.get_tokenizer()
        texts = ["this is a test", "I have pizza for lunch"]
        tokenized = tokenizer(
            text_target=texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=FRAMEWORK,
        )
        self.assertEqual(len(tokenized["input_ids"]), len(texts))
        self.assertEqual(len(tokenized["input_ids"][0]), max_length)
        self.assertEqual(len(tokenized["input_ids"][1]), max_length)
        self.assertEqual(len(tokenized["attention_mask"][0]), max_length)
        self.assertEqual(len(tokenized["attention_mask"][1]), max_length)
        self.assertEqual(len(tokenized["token_type_ids"][0]), max_length)
        self.assertEqual(len(tokenized["token_type_ids"][1]), max_length)

    def test_special_tokens_initialization(self):
        pass

    def test_special_tokens_initialization_with_non_empty_additional_special_tokens(self):
        pass

    def test_internal_consistency(self):
        pass

    def test_maximum_encoding_length_pair_input(self):
        pass

    def test_maximum_encoding_length_single_input(self):
        pass

    def test_pretokenized_inputs(self):
        pass

    def test_special_tokens_mask_input_pairs(self):
        pass

    def test_tokenizer_slow_store_full_signature(self):
        pass

    @slow
    def test_tokenizer_integration(self):
        pass
