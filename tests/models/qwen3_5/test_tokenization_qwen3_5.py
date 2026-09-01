# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers import Qwen3_5Tokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class Qwen3_5TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "Qwen/Qwen3.5-9B"
    tokenizer_class = Qwen3_5Tokenizer

    # Matches Qwen3_5Tokenizer.from_pretrained("Qwen/Qwen3.5-9B"): byte-level BPE with the Qwen pre-tokenizer
    # regex, so whitespace is carried into the tokens as "Ġ".
    integration_expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ', '9', '2', '0', '0', '0', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁæ´»çļĦ', 'çľŁè°Ľ', 'æĺ¯', 'Ċ', 'Hi', 'Ġ', 'ĠHello', 'Ċ', 'Hi', 'ĠĠ', 'ĠHello', 'ĊĊ', 'ĠĊĠĠĊ', 'ĠHello', 'Ċ', '<s', '>', 'Ċ', 'hi', '<s', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġà¸Ľà¸µ', 'ĠĠ', 'Ġ', 'ird', 'ĠĠ', 'Ġà¸Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_token_ids = [1919, 369, 264, 1228, 25677, 232, 198, 40, 557, 8950, 303, 220, 24, 17, 15, 15, 15, 11, 321, 411, 369, 30882, 933, 13, 198, 103815, 132339, 95761, 198, 12675, 220, 21251, 198, 12675, 256, 21251, 271, 46813, 21251, 198, 42589, 29, 198, 5834, 42589, 29, 17977, 198, 760, 2614, 886, 1220, 381, 9971, 19873, 25, 21251, 13, 198, 3850, 220, 2517, 321, 170827, 256, 220, 2517, 256, 149027, 198, 18103, 1204, 513, 488, 3604]  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"

    def test_checkpoint_special_tokens(self):
        # What a user actually gets from the checkpoint: the chat-tuned <|im_end|> as eos, with <|endoftext|>
        # kept for padding. Neither bos nor unk is defined, so generation must not assume them.
        tokenizer = self.get_tokenizer()

        self.assertEqual(tokenizer.eos_token, "<|im_end|>")
        self.assertEqual(tokenizer.pad_token, "<|endoftext|>")
        self.assertIsNone(tokenizer.bos_token)
        self.assertIsNone(tokenizer.unk_token)

    def test_class_default_special_tokens(self):
        # The class's own defaults, used when no tokenizer config supplies them: everything falls back to
        # <|endoftext|>, which is also the only entry in the default vocabulary.
        tokenizer = Qwen3_5Tokenizer()

        self.assertEqual(tokenizer.get_vocab(), {"<|endoftext|>": 0})
        self.assertEqual(tokenizer.unk_token, "<|endoftext|>")
        self.assertEqual(tokenizer.eos_token, "<|endoftext|>")
        self.assertEqual(tokenizer.pad_token, "<|endoftext|>")
        self.assertIsNone(tokenizer.bos_token)

    def test_no_prefix_space_is_added(self):
        # add_prefix_space defaults to False, so a leading word is not given a synthetic space.
        tokenizer = self.get_tokenizer()

        self.assertFalse(tokenizer.add_prefix_space)
        self.assertEqual(tokenizer.tokenize("test")[0], "test")

    def test_attention_mask_is_a_model_input(self):
        tokenizer = self.get_tokenizer()

        self.assertEqual(tokenizer.model_input_names, ["input_ids", "attention_mask"])
