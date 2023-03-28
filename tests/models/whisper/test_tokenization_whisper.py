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

from transformers.models.whisper import WhisperTokenizer, WhisperTokenizerFast
from transformers.models.whisper.tokenization_whisper import _find_longest_common_sequence
from transformers.testing_utils import slow

from ...test_tokenization_common import TokenizerTesterMixin


ES_CODE = 50262
EN_CODE = 50259
END_OF_TRANSCRIPT = 50257
START_OF_TRANSCRIPT = 50258
TRANSLATE = 50358
TRANSCRIBE = 50359
NOTIMESTAMPS = 50363


class WhisperTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = WhisperTokenizer
    rust_tokenizer_class = WhisperTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = False
    test_seq2seq = False

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
        self.assertEqual(self.get_tokenizer().vocab_size, 50258)

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

    def test_tokenizer_fast_store_full_signature(self):
        pass

    def test_special_tokens_initialization(self):
        # Whisper relies on specific additional special tokens, so we skip this
        # general test. In particular, this test loads fast tokenizer from slow
        # tokenizer, and the conversion uses prefix_tokens, where we reference
        # additional special tokens by specific indices, hence overriding the
        # list with less tokens leads to out of index error
        pass

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[50257, 50362, 41762, 364, 357, 36234, 1900, 355, 12972, 13165, 354, 12, 35636, 364, 290, 12972, 13165, 354, 12, 5310, 13363, 12, 4835, 8, 3769, 2276, 12, 29983, 45619, 357, 13246, 51, 11, 402, 11571, 12, 17, 11, 5564, 13246, 38586, 11, 16276, 44, 11, 4307, 346, 33, 861, 11, 16276, 7934, 23029, 329, 12068, 15417, 28491, 357, 32572, 52, 8, 290, 12068, 15417, 16588, 357, 32572, 38, 8, 351, 625, 3933, 10, 2181, 13363, 4981, 287, 1802, 10, 8950, 290, 2769, 48817, 1799, 1022, 449, 897, 11, 9485, 15884, 354, 290, 309, 22854, 37535, 13, 50256], [50257, 50362, 13246, 51, 318, 3562, 284, 662, 12, 27432, 2769, 8406, 4154, 282, 24612, 422, 9642, 9608, 276, 2420, 416, 26913, 21143, 319, 1111, 1364, 290, 826, 4732, 287, 477, 11685, 13, 50256], [50257, 50362, 464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 50256]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # noqa: E501
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding, model_name="openai/whisper-tiny.en", padding=False
        )

    def test_output_offsets(self):
        tokenizer = self.get_tokenizer()
        previous_sequence = [51492, 406, 3163, 1953, 466, 13, 51612, 51612]
        self.assertEqual(
            tokenizer.decode(previous_sequence, output_offsets=True),
            {
                "text": " not worth thinking about.",
                "offsets": [{"text": " not worth thinking about.", "timestamp": (22.56, 24.96)}],
            },
        )

        # Merge when the previous sequence is a suffix of the next sequence
        # fmt: off
        next_sequences_1 = [50364, 295, 6177, 3391, 11, 19817, 3337, 507, 307, 406, 3163, 1953, 466, 13, 50614, 50614, 2812, 9836, 14783, 390, 6263, 538, 257, 1359, 11, 8199, 6327, 1090, 322, 702, 7443, 13, 50834, 50257]
        # fmt: on
        self.assertEqual(
            tokenizer.decode(next_sequences_1, output_offsets=True),
            {
                "text": (
                    " of spectators, retrievality is not worth thinking about. His instant panic was followed by a"
                    " small, sharp blow high on his chest.<|endoftext|>"
                ),
                "offsets": [
                    {"text": " of spectators, retrievality is not worth thinking about.", "timestamp": (0.0, 5.0)},
                    {
                        "text": " His instant panic was followed by a small, sharp blow high on his chest.",
                        "timestamp": (5.0, 9.4),
                    },
                ],
            },
        )

    def test_find_longest_common_subsequence(self):
        previous_sequence = [1, 2, 3]
        next_sequence = [2, 3, 4, 5]
        merge = _find_longest_common_sequence([previous_sequence, next_sequence])
        self.assertEqual(merge, [1, 2, 3, 4, 5])

        # Now previous is larger than next.
        # We merge what we can and remove the extra right side of the left sequence
        previous_sequence = [1, 2, 3, 4, 5, 6, 7]
        next_sequence = [2, 3, 4, 5]
        merge = _find_longest_common_sequence([previous_sequence, next_sequence])
        self.assertEqual(merge, [1, 2, 3, 4, 5])

        # Nothing in common
        previous_sequence = [1, 2, 3]
        next_sequence = [4, 5, 6]
        merge = _find_longest_common_sequence([previous_sequence, next_sequence])
        self.assertEqual(merge, [1, 2, 3, 4, 5, 6])

        # Some errors in the overlap.
        # We take from previous on the left, from the next on the right of the overlap
        previous_sequence = [1, 2, 3, 4, 99]
        next_sequence = [2, 98, 4, 5, 6]
        merge = _find_longest_common_sequence([previous_sequence, next_sequence])
        self.assertEqual(merge, [1, 2, 3, 4, 5, 6])

        # We take from previous on the left, from the next on the right of the overlap
        previous_sequence = [1, 2, 99, 4, 5]
        next_sequence = [2, 3, 4, 98, 6]
        merge = _find_longest_common_sequence([previous_sequence, next_sequence])
        self.assertEqual(merge, [1, 2, 99, 4, 98, 6])

        # This works on 3 sequences
        seq1 = [1, 2, 3]
        seq2 = [2, 3, 4]
        seq3 = [3, 4, 5]
        merge = _find_longest_common_sequence([seq1, seq2, seq3])
        self.assertEqual(merge, [1, 2, 3, 4, 5])

        # This works on 3 sequences with errors
        seq1 = [1, 2, 3, 98, 5]
        seq2 = [2, 99, 4, 5, 6, 7]
        seq3 = [4, 97, 6, 7, 8]
        merge = _find_longest_common_sequence([seq1, seq2, seq3])
        self.assertEqual(merge, [1, 2, 3, 4, 5, 6, 7, 8])


class SpeechToTextTokenizerMultilinguialTest(unittest.TestCase):
    checkpoint_name = "openai/whisper-small.en"

    @classmethod
    def setUpClass(cls):
        cls.tokenizer: WhisperTokenizer = WhisperTokenizer.from_pretrained(cls.checkpoint_name)
        return cls

    def test_tokenizer_equivalence(self):
        text = "다람쥐 헌 쳇바퀴에 타고파"
        multilingual_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="korean")
        monolingual_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")

        monolingual_tokens = monolingual_tokenizer.encode(text, add_special_tokens=False)
        multilingual_tokens = multilingual_tokenizer.encode(text, add_special_tokens=False)

        assert monolingual_tokenizer.decode(monolingual_tokens) == text
        assert multilingual_tokenizer.decode(multilingual_tokens) == text
        assert len(monolingual_tokens) > len(multilingual_tokens)

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

        self.assertListEqual(monolingual_tokens, EXPECTED_ENG)
        self.assertListEqual(multilingual_tokens, EXPECTED_MULTI)

    def test_tokenizer_special(self):
        multilingual_tokenizer = WhisperTokenizer.from_pretrained(
            "openai/whisper-tiny", language="english", task="transcribe"
        )
        text = "Hey! How are you feeling? J'ai l'impression que 郷さん est prêt"

        multilingual_tokens = multilingual_tokenizer.encode(text)

        # fmt: off
        # format: <|startoftranscript|> <|lang-id|> <|task|> <|notimestamps|> ... transcription ids ... <|endoftext|>
        EXPECTED_MULTI = [
            START_OF_TRANSCRIPT, EN_CODE, TRANSCRIBE, NOTIMESTAMPS, 7057, 0, 1012, 366, 291,
            2633, 30, 508, 6, 1301, 287, 6, 36107, 631, 220, 11178,
            115, 15567, 871, 44393, END_OF_TRANSCRIPT
        ]
        EXPECTED_SPECIAL_TEXT = (
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>Hey! How are you feeling? "
            "J'ai l'impression que 郷さん est prêt<|endoftext|>"
        )
        # fmt: on

        self.assertListEqual(multilingual_tokens, EXPECTED_MULTI)

        special_transcript = multilingual_tokenizer.decode(multilingual_tokens, skip_special_tokens=False)
        self.assertEqual(special_transcript, EXPECTED_SPECIAL_TEXT)

        transcript = multilingual_tokenizer.decode(multilingual_tokens, skip_special_tokens=True)
        self.assertEqual(transcript, text)

    def test_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, 50257)

    # Copied from transformers.tests.speech_to_test.test_tokenization_speech_to_text.py
    def test_tokenizer_decode_ignores_language_codes(self):
        self.assertIn(ES_CODE, self.tokenizer.all_special_ids)
        generated_ids = [ES_CODE, 4, 1601, 47, 7647, 2]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        expected_spanish = self.tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        self.assertEqual(result, expected_spanish)
        self.assertNotIn(self.tokenizer.eos_token, result)

    def test_batch_encoding(self):
        multilingual_tokenizer = WhisperTokenizer.from_pretrained(
            "openai/whisper-tiny", language="spanish", task="translate"
        )
        batch = ["El gato ", "El gato se sentó"]
        batch_output = multilingual_tokenizer.batch_encode_plus(batch, padding=True).input_ids

        # fmt: off
        EXPECTED_MULTI = [
            [START_OF_TRANSCRIPT, ES_CODE, TRANSLATE, NOTIMESTAMPS, 17356, 290, 2513, 220,
             END_OF_TRANSCRIPT, END_OF_TRANSCRIPT, END_OF_TRANSCRIPT],
            [START_OF_TRANSCRIPT, ES_CODE, TRANSLATE, NOTIMESTAMPS, 17356, 290, 2513, 369,
             2279, 812, END_OF_TRANSCRIPT]
        ]
        # fmt: on

        self.assertListEqual(batch_output, EXPECTED_MULTI)

    def test_set_prefix_tokens(self):
        multilingual_tokenizer = WhisperTokenizer.from_pretrained(
            "openai/whisper-tiny", language="spanish", task="translate"
        )

        # change the language prefix token from Spanish to English
        multilingual_tokenizer.set_prefix_tokens(language="english")

        batch = ["the cat", "the cat sat"]
        batch_output = multilingual_tokenizer.batch_encode_plus(batch, padding=True).input_ids

        # fmt: off
        EXPECTED_MULTI = [
            [START_OF_TRANSCRIPT, EN_CODE, TRANSLATE, NOTIMESTAMPS, 3322, 3857,
             END_OF_TRANSCRIPT, END_OF_TRANSCRIPT],
            [START_OF_TRANSCRIPT, EN_CODE, TRANSLATE, NOTIMESTAMPS, 3322, 3857,
             3227, END_OF_TRANSCRIPT]
        ]
        # fmt: on

        self.assertListEqual(batch_output, EXPECTED_MULTI)

    def test_batch_encoding_decoding(self):
        multilingual_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="spanish")
        batch = ["hola güey", "que onda"]
        batch_encoding = multilingual_tokenizer.batch_encode_plus(batch, padding=True).input_ids
        transcription = multilingual_tokenizer.batch_decode(batch_encoding, skip_special_tokens=True)
        self.assertListEqual(batch, transcription)

    def test_offset_decoding(self):
        multilingual_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        # fmt: off
        INPUT_TOKENS = [
            50258, 50259, 50359, 50364, 441, 1857, 4174, 11, 5242, 366,
            257, 1333, 295, 493, 2794, 2287, 293, 12018, 14880, 11,
            293, 25730, 311, 454, 34152, 4496, 904, 50724, 50724, 366,
            382, 4048, 382, 257, 361, 18459, 13065, 13, 2221, 13,
            7145, 74, 325, 38756, 311, 29822, 7563, 412, 472, 709,
            294, 264, 51122, 51122, 912, 636, 300, 2221, 13, 2741,
            5767, 1143, 281, 7319, 702, 7798, 13, 400, 2221, 13,
            2619, 4004, 811, 2709, 702, 51449, 51449, 50257
        ]
        # fmt: on
        output = multilingual_tokenizer.decode(INPUT_TOKENS, output_offsets=True)["offsets"]

        self.assertEqual(
            output,
            [
                {
                    "text": (
                        " Lennils, pictures are a sort of upguards and atom paintings, and Mason's exquisite idles"
                    ),
                    "timestamp": (0.0, 7.2),
                },
                {
                    "text": (
                        " are as national as a jingo poem. Mr. Birkut Foster's landscapes smile at one much in the"
                    ),
                    "timestamp": (7.2, 15.16),
                },
                {
                    "text": " same way that Mr. Carker used to flash his teeth. And Mr. John Colier gives his",
                    "timestamp": (15.16, 21.7),
                },
            ],
        )
        # test `decode_with_offsets`
        output = multilingual_tokenizer.decode(INPUT_TOKENS, decode_with_timestamps=True)
        self.assertEqual(
            output,
            "<|startoftranscript|><|en|><|transcribe|><|0.00|> Lennils, pictures are a sort of upguards and atom"
            " paintings, and Mason's exquisite idles<|7.20|><|7.20|> are as national as a jingo poem. Mr. Birkut"
            " Foster's landscapes smile at one much in the<|15.16|><|15.16|> same way that Mr. Carker used to flash"
            " his teeth. And Mr. John Colier gives his<|21.70|><|21.70|><|endoftext|>",
        )
        # test a single sequence with timestamps
        # fmt: off
        INPUT_TOKENS = [
            50364, 441, 1857, 4174, 11, 5242, 366,
            257, 1333, 295, 493, 2794, 2287, 293, 12018, 14880, 11,
            293, 25730, 311, 454, 34152, 4496, 904, 50724
        ]
        # fmt: on

        output = multilingual_tokenizer.decode(INPUT_TOKENS, output_offsets=True)["offsets"]
        self.assertEqual(
            output[0],
            {
                "text": " Lennils, pictures are a sort of upguards and atom paintings, and Mason's exquisite idles",
                "timestamp": (0.0, 7.2),
            },
        )

        # test a sequence without a single timestamps
        # fmt: off
        INPUT_TOKENS = [
            441, 1857, 4174, 11, 5242, 366,
            257, 1333, 295, 493, 2794, 2287, 293, 12018, 14880, 11,
            293, 25730, 311, 454, 34152, 4496, 904, 50724
        ]
        # fmt: on

        output = multilingual_tokenizer.decode(INPUT_TOKENS, output_offsets=True)["offsets"]
        self.assertEqual(output, [])
