# coding=utf-8
# Copyright 2019 Hugging Face inc.
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

import os
import unittest

from transformers import DebertaV2Tokenizer
from transformers.testing_utils import require_sentencepiece, require_tokenizers

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/spiece.model")


@require_sentencepiece
@require_tokenizers
class DebertaV2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = DebertaV2Tokenizer
    rust_tokenizer_class = None
    test_rust_tokenizer = False
    test_sentencepiece = True
    test_sentencepiece_ignore_case = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = DebertaV2Tokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

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

    def test_full_tokenizer(self):
        tokenizer = DebertaV2Tokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁", "[UNK]", "his", "▁is", "▁a", "▁test"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [13, 1, 4398, 25, 21, 1289])

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        # fmt: off
        self.assertListEqual(
            tokens,
            ["▁", "[UNK]", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "[UNK]", "."],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [13, 1, 23, 386, 19, 561, 3050, 15, 17, 48, 25, 8256, 18, 1, 9])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            ["▁", "<unk>", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "."],
        )
        # fmt: on

    def test_sequence_builders(self):
        tokenizer = DebertaV2Tokenizer(SAMPLE_VOCAB)

        text = tokenizer.encode("sequence builders")
        text_2 = tokenizer.encode("multi-sequence build")

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        self.assertEqual([tokenizer.cls_token_id] + text + [tokenizer.sep_token_id], encoded_sentence)
        self.assertEqual(
            [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [tokenizer.sep_token_id],
            encoded_pair,
        )

    def test_tokenizer_integration(self):
        tokenizer_classes = [self.tokenizer_class]
        if self.test_rust_tokenizer:
            tokenizer_classes.append(self.rust_tokenizer_class)

        for tokenizer_class in tokenizer_classes:
            tokenizer = tokenizer_class.from_pretrained("microsoft/deberta-xlarge-v2")

            sequences = [
                [
                    "DeBERTa: Decoding-enhanced BERT with Disentangled Attention",
                    "DeBERTa: Decoding-enhanced BERT with Disentangled Attention",
                ],
                [
                    "Recent progress in pre-trained neural language models has significantly improved the performance of many natural language processing (NLP) tasks.",
                    "DeBERTa: Decoding-enhanced BERT with Disentangled Attention",
                ],
                [
                    "In this paper we propose a new model architecture DeBERTa",
                    "DeBERTa: Decoding-enhanced BERT with Disentangled Attention",
                ],
            ]

            encoding = tokenizer(sequences, padding=True)
            decoded_sequences = [tokenizer.decode(seq, skip_special_tokens=True) for seq in encoding["input_ids"]]

            # fmt: off
            expected_encoding = {
                'input_ids': [
                    [1, 1804, 69418, 191, 43, 117056, 18, 44596, 448, 37132, 19, 8655, 10625, 69860, 21149, 2, 1804, 69418, 191, 43, 117056, 18, 44596, 448, 37132, 19, 8655, 10625, 69860, 21149, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 9755, 1944, 11, 1053, 18, 16899, 12730, 1072, 1506, 45, 2497, 2510, 5, 610, 9, 127, 699, 1072, 2101, 36, 99388, 53, 2930, 4, 2, 1804, 69418, 191, 43, 117056, 18, 44596, 448, 37132, 19, 8655, 10625, 69860, 21149, 2],
                    [1, 84, 32, 778, 42, 9441, 10, 94, 735, 3372, 1804, 69418, 191, 2, 1804, 69418, 191, 43, 117056, 18, 44596, 448, 37132, 19, 8655, 10625, 69860, 21149, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                'token_type_ids': [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                'attention_mask': [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ]
            }

            expected_decoded_sequences = [
                'DeBERTa: Decoding-enhanced BERT with Disentangled Attention DeBERTa: Decoding-enhanced BERT with Disentangled Attention',
                'Recent progress in pre-trained neural language models has significantly improved the performance of many natural language processing (NLP) tasks. DeBERTa: Decoding-enhanced BERT with Disentangled Attention',
                'In this paper we propose a new model architecture DeBERTa DeBERTa: Decoding-enhanced BERT with Disentangled Attention'
            ]
            # fmt: on

            self.assertDictEqual(encoding.data, expected_encoding)

            for expected, decoded in zip(expected_decoded_sequences, decoded_sequences):
                self.assertEqual(expected, decoded)
