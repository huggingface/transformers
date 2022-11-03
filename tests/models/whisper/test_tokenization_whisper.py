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

from transformers.models.whisper import WhisperTokenizer
from transformers.testing_utils import slow

from ...test_tokenization_common import TokenizerTesterMixin


EN_CODE = 50258
ES_CODE = 50256


class WhisperTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = WhisperTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = False

    def setUp(self):
        super().setUp()
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        tokenizer.pad_token_id = 50256
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.save_pretrained(self.tmpdirname)

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
        self.assertEqual(vocab_keys[-1], "<|notimestamps|>")
        self.assertEqual(len(vocab_keys), 50364)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 50257)

    def test_full_tokenizer(self):
        tokenizer = WhisperTokenizer.from_pretrained(self.tmpdirname)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["This", "Ġis", "Ġa", "Ġ", "test"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [5723, 307, 257, 220, 31636],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            # fmt: off
            ['I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ9', '2000', ',', 'Ġand', 'Ġ', 'this', 'Ġis', 'Ġfals', 'Ã©', '.'],
            # fmt: on
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [40, 390, 4232, 294, 1722, 25743, 11, 293, 220, 11176, 307, 16720, 526, 13])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            # fmt: off
            ['I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ9', '2000', ',', 'Ġand', 'Ġ', 'this', 'Ġis', 'Ġfals', 'Ã©', '.'],
            # fmt: on
        )

    def test_tokenizer_slow_store_full_signature(self):
        pass

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[41762, 364, 357, 36234, 1900, 355, 12972, 13165, 354, 12, 35636, 364, 290, 12972, 13165, 354, 12, 5310, 13363, 12, 4835, 8, 3769, 2276, 12, 29983, 45619, 357, 13246, 51, 11, 402, 11571, 12, 17, 11, 5564, 13246, 38586, 11, 16276, 44, 11, 4307, 346, 33, 861, 11, 16276, 7934, 23029, 329, 12068, 15417, 28491, 357, 32572, 52, 8, 290, 12068, 15417, 16588, 357, 32572, 38, 8, 351, 625, 3933, 10, 2181, 13363, 4981, 287, 1802, 10, 8950, 290, 2769, 48817, 1799, 1022, 449, 897, 11, 9485, 15884, 354, 290, 309, 22854, 37535, 13], [13246, 51, 318, 3562, 284, 662, 12, 27432, 2769, 8406, 4154, 282, 24612, 422, 9642, 9608, 276, 2420, 416, 26913, 21143, 319, 1111, 1364, 290, 826, 4732, 287, 477, 11685, 13], [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # noqa: E501
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding, model_name="openai/whisper-tiny.en", padding=False
        )


class SpeechToTextTokenizerMultilinguialTest(unittest.TestCase):
    checkpoint_name = "openai/whisper-small.en"

    transcript = (
        "'<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>  Nor is Mr. Quilters manner less interesting"
        " than his matter.<|endoftext|>'"
    )
    clean_transcript = "  Nor is Mr. Quilters manner less interesting than his matter."
    french_text = "Bonjour! Il me semble que Mrs Quilters n'était pas présente"

    @classmethod
    def setUpClass(cls):
        cls.tokenizer: WhisperTokenizer = WhisperTokenizer.from_pretrained(cls.checkpoint_name)
        return cls

    def test_tokenizer_equivalence(self):
        text = "다람쥐 헌 쳇바퀴에 타고파"
        multilingual_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="ko")
        gpt2_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")

        gpt2_tokens = gpt2_tokenizer.encode(text)
        multilingual_tokens = multilingual_tokenizer.encode(text)

        assert gpt2_tokenizer.decode(gpt2_tokens) == text
        assert multilingual_tokenizer.decode(multilingual_tokens) == text
        assert len(gpt2_tokens) > len(multilingual_tokens)

        # fmt: off
        EXPECTED_ENG = [
            46695, 97, 167, 252, 234, 168, 98, 238, 220, 169,
            245, 234, 23821, 111, 229, 167, 108, 242, 169, 222,
            112, 168, 245, 238, 220, 169, 225, 222, 166, 111,
            254, 169, 234, 234
        ]
        EXPECTED_MULTI = [
            9835, 22855, 168, 98, 238, 13431, 234, 43517, 229, 47053,
            169, 222, 19086, 19840, 1313, 17974
        ]
        # fmt: on

        self.assertListEqual(gpt2_tokens, EXPECTED_ENG)
        self.assertListEqual(multilingual_tokens, EXPECTED_MULTI)

    def test_tokenizer_special(self):
        multilingual_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
        text = "<|startoftranscript|>Hey! How are you feeling? J'ai l'impression que 郷さん est prêt<|endoftext|>"

        multilingual_tokens = multilingual_tokenizer.encode(text)

        # fmt: off
        EXPECTED_MULTI = [
            50257, 10814, 0, 1374, 389, 345, 4203, 30, 449, 6,
            1872, 300, 6, 11011, 2234, 8358, 16268, 225, 115, 43357,
            22174, 1556, 778, 25792, 83, 50256
        ]
        # fmt: on

        self.assertListEqual(multilingual_tokens, EXPECTED_MULTI)

        self.assertEqual(text, multilingual_tokenizer.decode(multilingual_tokens))

        transcript = multilingual_tokenizer.decode(multilingual_tokens, skip_special_tokens=True)

        EXPECTED_JAP = "Hey! How are you feeling? J'ai l'impression que 郷さん est prêt"
        self.assertEqual(transcript, EXPECTED_JAP)

    def test_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, 50257)

    def test_tokenizer_decode_ignores_language_codes(self):
        self.assertIn(ES_CODE, self.tokenizer.all_special_ids)
        generated_ids = [ES_CODE, 4, 1601, 47, 7647, 2]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        expected_spanish = self.tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        self.assertEqual(result, expected_spanish)
        self.assertNotIn(self.tokenizer.eos_token, result)

    def test_batch_encoding(self):
        multilingual_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
        batch = ["<|en|><|notimestamps|>", "<|en|><|notimestamps|>I am sure that"]
        batch_output = multilingual_tokenizer.batch_encode_plus(batch, padding=True).input_ids

        # fmt: off
        EXPECTED_MULTI = [
            [50258, 50362, 50256, 50256, 50256, 50256],
            [50258, 50362, 40, 716, 1654, 326]
        ]
        # fmt: on

        self.assertListEqual(batch_output, EXPECTED_MULTI)
