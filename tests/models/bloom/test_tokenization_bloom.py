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

from transformers import TokenizersBackend
from transformers.testing_utils import require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class BloomTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "bigscience/tokenizer"
    slow_tokenizer_class = None
    rust_tokenizer_class = TokenizersBackend
    tokenizer_class = TokenizersBackend
    test_slow_tokenizer = False
    from_pretrained_vocab_key = "tokenizer_file"
    special_tokens_map = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"}

    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ9', '2000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.Ċ', 'çĶŁæ´»çļĦ', 'çľŁ', 'è°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', 'Ġ', 'ĠHello', 'Ċ', 'Hi', 'ĠĠ', 'ĠHello', 'ĊĊ', 'ĠĊ', 'ĠĠĊ', 'ĠHello', 'Ċ', '<s>', 'Ċ', 'hi', '<s>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġenc', 'od', 'ed:', 'ĠHello', '.Ċ', 'But', 'Ġir', 'd', 'Ġand', 'Ġà¸', 'Ľ', 'à¸µ', 'ĠĠ', 'Ġir', 'd', 'ĠĠ', 'Ġà¸', 'Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_token_ids = [6168, 632, 267, 4006, 189, 44, 1620, 34181, 361, 1575, 14739, 15, 530, 1119, 632, 31684, 311, 336, 71167, 4137, 1927, 239, 644, 189, 30050, 210, 86153, 189, 30050, 250, 86153, 603, 5306, 33249, 86153, 189, 1, 189, 2807, 1, 51596, 189, 2175, 6747, 5148, 3403, 722, 34975, 2681, 532, 29315, 86153, 336, 6475, 2881, 71, 530, 44381, 239, 105442, 250, 2881, 71, 250, 44381, 232, 189, 40440, 4143, 1306, 1152, 12491]  # fmt: skip

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        tokenizer = TokenizersBackend.from_pretrained("bigscience/tokenizer")
        tokenizer.save_pretrained(cls.tmpdirname)
        cls.tokenizers_list = [(cls.rust_tokenizer_class, cls.tmpdirname, {})]

    def test_encodings_from_sample_data(self):
        """
        Assert that the created tokens are the same than the hard-coded ones
        """
        tokenizer = self.get_tokenizer()

        INPUT_SENTENCES = ["The quick brown fox</s>", "jumps over the lazy dog</s>"]
        TARGET_TOKENS = [[2175, 23714, 73173, 144252, 2], [77, 132619, 3478, 368, 109586, 35433, 2]]

        computed_tokens = tokenizer(INPUT_SENTENCES)["input_ids"]
        self.assertListEqual(TARGET_TOKENS, computed_tokens)

        decoded_tokens = tokenizer.decode(computed_tokens)
        self.assertListEqual(decoded_tokens, INPUT_SENTENCES)

    def test_padding(self, max_length=6):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.get_tokenizer(pretrained_name, **kwargs)
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
                    tokenizer_r(s, max_length=max_length)

                    tokenizer_r(s2, max_length=max_length)
                    tokenizer_r.encode(p, max_length=max_length)
                    tokenizer_r(p2, max_length=max_length)
                except ValueError:
                    self.fail("Bloom Tokenizer should be able to deal with padding")

                tokenizer_r.pad_token = None  # Hotfixing padding = None
                self.assertRaises(ValueError, tokenizer_r.encode, s, max_length=max_length, padding="max_length")

                # Simple input
                self.assertRaises(ValueError, tokenizer_r, s, max_length=max_length, padding="max_length")

                # Simple input
                self.assertRaises(
                    ValueError,
                    tokenizer_r,
                    s2,
                    max_length=max_length,
                    padding="max_length",
                )

                # Pair input
                self.assertRaises(ValueError, tokenizer_r.encode, p, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(ValueError, tokenizer_r, p, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(
                    ValueError,
                    tokenizer_r,
                    p2,
                    max_length=max_length,
                    padding="max_length",
                )

    def test_encodings_from_xnli_dataset(self):
        """
        Tests the tokenizer downloaded from here:
            - https://huggingface.co/bigscience/tokenizer/
        """
        tokenizer = self.get_tokenizer()
        ds = load_dataset("facebook/xnli", "all_languages", split="test", streaming=True)

        sample_data = next(iter(ds))["premise"]  # pick up one data
        input_text = list(sample_data.values())

        output_tokens = list(map(tokenizer.encode, input_text))
        predicted_text = [tokenizer.decode(x, clean_up_tokenization_spaces=False) for x in output_tokens]
        self.assertListEqual(predicted_text, input_text)

    @slow
    def test_save_and_load_tokenizer(self):
        return super().test_save_and_load_tokenizer()
