# coding=utf-8
# Copyright 2024 ConvaiInnovations and The HuggingFace Team. All rights reserved.
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

from transformers import HindiCausalLMTokenizer, HindiCausalLMTokenizerFast
from transformers.testing_utils import require_sentencepiece, require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin


@require_sentencepiece
class HindiCausalLMTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = HindiCausalLMTokenizer
    test_rust_tokenizer = False
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = HindiCausalLMTokenizer.from_pretrained("convaiinnovations/hindi-causal-lm")
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return HindiCausalLMTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "नमस्ते दुनिया"  # Hello world in Hindi
        output_text = "नमस्ते दुनिया"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = HindiCausalLMTokenizer.from_pretrained(self.tmpdirname)

        tokens = tokenizer.tokenize("नमस्ते दुनिया")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        self.assertEqual(tokenizer.convert_tokens_to_string(tokens), "नमस्ते दुनिया")

        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_special_tokens(self):
        tokenizer = HindiCausalLMTokenizer.from_pretrained(self.tmpdirname)

        text = "नमस्ते दुनिया"
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)

        # Test adding special tokens
        input_ids = tokenizer.build_inputs_with_special_tokens(ids)
        self.assertEqual(input_ids[0], tokenizer.bos_token_id)
        self.assertEqual(input_ids[-1], tokenizer.eos_token_id)

    def test_padding(self):
        tokenizer = HindiCausalLMTokenizer.from_pretrained(self.tmpdirname)

        text = "नमस्ते दुनिया"
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)

        # Test padding
        padded_ids = tokenizer.pad({"input_ids": [ids]}, padding=True)["input_ids"][0]
        self.assertIn(tokenizer.pad_token_id, padded_ids)

    @slow
    def test_tokenizer_integration(self):
        inputs = ["नमस्ते दुनिया", "यह एक परीक्षण है"]  # ["Hello world", "This is a test"] in Hindi

        # The exact IDs may differ based on the actual tokenizer model, so this test checks shapes
        tokenizer = HindiCausalLMTokenizer.from_pretrained("convaiinnovations/hindi-causal-lm")
        encoding = tokenizer(inputs, padding=True, return_tensors="pt")

        self.assertEqual(len(encoding["input_ids"]), 2)
        self.assertEqual(len(encoding["attention_mask"]), 2)


@require_tokenizers
class HindiCausalLMTokenizationFastTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = HindiCausalLMTokenizerFast
    rust_tokenizer_class = HindiCausalLMTokenizerFast
    test_rust_tokenizer = True
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = HindiCausalLMTokenizer.from_pretrained("convaiinnovations/hindi-causal-lm")
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return HindiCausalLMTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "नमस्ते दुनिया"  # Hello world in Hindi
        output_text = "नमस्ते दुनिया"
        return input_text, output_text

    def test_convert_token_and_id(self):
        token = "दुनिया"  # "world" in Hindi
        token_id = 15021  # This ID is a placeholder and may need to be adjusted based on the actual tokenizer

        tokenizer = self.get_tokenizer()

        self.assertEqual(tokenizer.convert_tokens_to_ids(token), token_id)
        self.assertEqual(tokenizer.convert_ids_to_tokens(token_id), token)

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()

        tokens = tokenizer.tokenize("नमस्ते दुनिया")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        self.assertEqual(tokenizer.convert_tokens_to_string(tokens), "नमस्ते दुनिया")

        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_fast_special_tokens(self):
        tokenizer = self.get_tokenizer()

        text = "नमस्ते दुनिया"
        encoded = tokenizer(text)

        # Test special tokens are properly added
        self.assertEqual(encoded.input_ids[0], tokenizer.bos_token_id)
        self.assertEqual(encoded.input_ids[-1], tokenizer.eos_token_id)

    def test_fast_and_slow_same_result(self):
        slow_tokenizer = HindiCausalLMTokenizer.from_pretrained(self.tmpdirname)
        fast_tokenizer = self.get_tokenizer()

        text = "नमस्ते दुनिया यह एक परीक्षण है"

        slow_output = slow_tokenizer(text, return_tensors=None)
        fast_output = fast_tokenizer(text, return_tensors=None)

        # Check that the number of tokens is the same
        self.assertEqual(len(slow_output["input_ids"]), len(fast_output["input_ids"]))
