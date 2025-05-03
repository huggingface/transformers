# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import numpy as np

from transformers.models.dia import DiaTokenizer, DiaTokenizerFast
from transformers.models.dia.tokenization_dia import _combine_tokens_into_words, _find_longest_common_sequence
from transformers.testing_utils import require_flax, require_tf, require_torch, slow

from ...test_tokenization_common import TokenizerTesterMixin


ES_CODE = 50262
EN_CODE = 50259
END_OF_TRANSCRIPT = 50257
START_OF_TRANSCRIPT = 50258
TRANSLATE = 50358
TRANSCRIBE = 50359
NOTIMESTAMPS = 50363


class DiaTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "nari-labs/Dia-1.6B"
    tokenizer_class = DiaTokenizer
    rust_tokenizer_class = DiaTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = False
    test_seq2seq = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tokenizer = DiaTokenizer.from_pretrained("nari-labs/Dia-1.6B")
        tokenizer.pad_token_id = 50256
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.save_pretrained(cls.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "Where"
        token_id = 14436

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "!")
        self.assertEqual(vocab_keys[1], '"')
        self.assertEqual(vocab_keys[-1], "<|30.00|>")
        self.assertEqual(len(vocab_keys), 51865)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 50258)

    def test_full_tokenizer(self):
        tokenizer = DiaTokenizer.from_pretrained(self.tmpdirname)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["This", "Ġis", "Ġa", "Ġtest"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [5723, 307, 257, 1500],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            ["I", "Ġwas", "Ġborn", "Ġin", "Ġ9", "2000", ",", "Ġand", "Ġthis", "Ġis", "Ġfals", "Ã©", "."],  # fmt: skip
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [40, 390, 4232, 294, 1722, 25743, 11, 293, 341, 307, 16720, 526, 13])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            ["I", "Ġwas", "Ġborn", "Ġin", "Ġ9", "2000", ",", "Ġand", "Ġthis", "Ġis", "Ġfals", "Ã©", "."],  # fmt: skip
        )

    @unittest.skip
    def test_tokenizer_slow_store_full_signature(self):
        pass

    @unittest.skip
    def test_tokenizer_fast_store_full_signature(self):
        pass

    @slow
    def test_tokenizer_integration(self):
        expected_encoding = {'input_ids': [[50257, 50362, 41762, 364, 357, 36234, 1900, 355, 12972, 13165, 354, 12, 35636, 364, 290, 12972, 13165, 354, 12, 5310, 13363, 12, 4835, 8, 3769, 2276, 12, 29983, 45619, 357, 13246, 51, 11, 402, 11571, 12, 17, 11, 5564, 13246, 38586, 11, 16276, 44, 11, 4307, 346, 33, 861, 11, 16276, 7934, 23029, 329, 12068, 15417, 28491, 357, 32572, 52, 8, 290, 12068, 15417, 16588, 357, 32572, 38, 8, 351, 625, 3933, 10, 2181, 13363, 4981, 287, 1802, 10, 8950, 290, 2769, 48817, 1799, 1022, 449, 897, 11, 9485, 15884, 354, 290, 309, 22854, 37535, 13, 50256], [50257, 50362, 13246, 51, 318, 3562, 284, 662, 12, 27432, 2769, 8406, 4154, 282, 24612, 422, 9642, 9608, 276, 2420, 416, 26913, 21143, 319, 1111, 1364, 290, 826, 4732, 287, 477, 11685, 13, 50256], [50257, 50362, 464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 50256]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # fmt: skip

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding, model_name="nari-labs/Dia-1.6B.en", padding=False
        )
