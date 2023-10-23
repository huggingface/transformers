# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from datasets import load_dataset

from transformers import BloomTokenizerFast
from transformers.testing_utils import require_jinja, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class BloomTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    slow_tokenizer_class = None
    rust_tokenizer_class = BloomTokenizerFast
    tokenizer_class = BloomTokenizerFast
    test_rust_tokenizer = True
    test_slow_tokenizer = False
    from_pretrained_vocab_key = "tokenizer_file"
    special_tokens_map = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"}

    def setUp(self):
        super().setUp()
        tokenizer = BloomTokenizerFast.from_pretrained("bigscience/tokenizer")
        tokenizer.save_pretrained(self.tmpdirname)

    def get_rust_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return BloomTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    @unittest.skip("This needs a slow tokenizer. Bloom does not have one!")
    def test_encode_decode_with_spaces(self):
        return

    def test_encodings_from_sample_data(self):
        """
        Assert that the created tokens are the same than the hard-coded ones
        """
        tokenizer = self.get_rust_tokenizer()

        INPUT_SENTENCES = ["The quick brown fox</s>", "jumps over the lazy dog</s>"]
        TARGET_TOKENS = [[2175, 23714, 73173, 144252, 2], [77, 132619, 3478, 368, 109586, 35433, 2]]

        computed_tokens = tokenizer.batch_encode_plus(INPUT_SENTENCES)["input_ids"]
        self.assertListEqual(TARGET_TOKENS, computed_tokens)

        decoded_tokens = tokenizer.batch_decode(computed_tokens)
        self.assertListEqual(decoded_tokens, INPUT_SENTENCES)

    def test_padding(self, max_length=6):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                # tokenizer_r.pad_token = None # Hotfixing padding = None
                # Simple input
                s = "This is a simple input"
                s2 = ["This is a simple input 1", "This is a simple input 2"]
                p = ("This is a simple input", "This is a pair")
                p2 = [
                    ("This is a simple input 1", "This is a simple input 2"),
                    ("This is a simple pair 1", "This is a simple pair 2"),
                ]

                # Simple input tests
                try:
                    tokenizer_r.encode(s, max_length=max_length)
                    tokenizer_r.encode_plus(s, max_length=max_length)

                    tokenizer_r.batch_encode_plus(s2, max_length=max_length)
                    tokenizer_r.encode(p, max_length=max_length)
                    tokenizer_r.batch_encode_plus(p2, max_length=max_length)
                except ValueError:
                    self.fail("Bloom Tokenizer should be able to deal with padding")

                tokenizer_r.pad_token = None  # Hotfixing padding = None
                self.assertRaises(ValueError, tokenizer_r.encode, s, max_length=max_length, padding="max_length")

                # Simple input
                self.assertRaises(ValueError, tokenizer_r.encode_plus, s, max_length=max_length, padding="max_length")

                # Simple input
                self.assertRaises(
                    ValueError,
                    tokenizer_r.batch_encode_plus,
                    s2,
                    max_length=max_length,
                    padding="max_length",
                )

                # Pair input
                self.assertRaises(ValueError, tokenizer_r.encode, p, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(ValueError, tokenizer_r.encode_plus, p, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(
                    ValueError,
                    tokenizer_r.batch_encode_plus,
                    p2,
                    max_length=max_length,
                    padding="max_length",
                )

    def test_encodings_from_xnli_dataset(self):
        """
        Tests the tokenizer downloaded from here:
            - https://huggingface.co/bigscience/tokenizer/
        """
        tokenizer = self.get_rust_tokenizer()
        ds = load_dataset("xnli", "all_languages", split="test", streaming=True)

        sample_data = next(iter(ds))["premise"]  # pick up one data
        input_text = list(sample_data.values())

        output_tokens = list(map(tokenizer.encode, input_text))
        predicted_text = [tokenizer.decode(x, clean_up_tokenization_spaces=False) for x in output_tokens]
        self.assertListEqual(predicted_text, input_text)

    def test_pretrained_model_lists(self):
        # The test has to be overriden because BLOOM uses ALiBi positional embeddings that does not have
        # any sequence length constraints. This test of the parent class will fail since it relies on the
        # maximum sequence length of the positoonal embeddings.
        self.assertGreaterEqual(len(self.tokenizer_class.pretrained_vocab_files_map), 1)
        self.assertGreaterEqual(len(list(self.tokenizer_class.pretrained_vocab_files_map.values())[0]), 1)

    @require_jinja
    def test_tokenization_for_chat(self):
        tokenizer = self.get_rust_tokenizer()
        test_chats = [
            [{"role": "system", "content": "You are a helpful chatbot."}, {"role": "user", "content": "Hello!"}],
            [
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Nice to meet you."},
            ],
            [{"role": "assistant", "content": "Nice to meet you."}, {"role": "user", "content": "Hello!"}],
        ]
        tokenized_chats = [tokenizer.apply_chat_template(test_chat) for test_chat in test_chats]
        expected_tokens = [
            [5448, 1306, 267, 66799, 44799, 37143, 17, 2, 59414, 4, 2],
            [5448, 1306, 267, 66799, 44799, 37143, 17, 2, 59414, 4, 2, 229126, 427, 11890, 1152, 17, 2],
            [229126, 427, 11890, 1152, 17, 2, 59414, 4, 2],
        ]
        for tokenized_chat, expected_tokens in zip(tokenized_chats, expected_tokens):
            self.assertListEqual(tokenized_chat, expected_tokens)

    def test_add_prefix_space_fast(self):
        tokenizer_w_prefix = self.get_rust_tokenizer(add_prefix_space=True)
        tokenizer_wo_prefix = self.get_rust_tokenizer(add_prefix_space=False)
        tokens_w_prefix = tokenizer_w_prefix.tokenize("Hey")
        tokens_wo_prefix = tokenizer_wo_prefix.tokenize("Hey")
        self.assertNotEqual(tokens_w_prefix, tokens_wo_prefix)
