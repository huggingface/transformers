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

from transformers import AlbertTokenizer, AlbertTokenizerFast
from transformers.testing_utils import require_sentencepiece, require_tokenizers, slow

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/spiece.model")


@require_sentencepiece
@require_tokenizers
class AlbertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = AlbertTokenizer
    rust_tokenizer_class = AlbertTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True
    test_sentencepiece_ignore_case = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = AlbertTokenizer(SAMPLE_VOCAB)
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
        tokenizer = AlbertTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁this", "▁is", "▁a", "▁test"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [48, 25, 21, 1289])

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens, ["▁i", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "é", "."]
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [31, 23, 386, 19, 561, 3050, 15, 17, 48, 25, 8256, 18, 1, 9])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            ["▁i", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "."],
        )

    def test_sequence_builders(self):
        tokenizer = AlbertTokenizer(SAMPLE_VOCAB)

        text = tokenizer.encode("sequence builders")
        text_2 = tokenizer.encode("multi-sequence build")

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [
            tokenizer.sep_token_id
        ]

    @slow
    def test_tokenizer_integration(self):
        tokenizer_classes = [self.tokenizer_class]
        if self.test_rust_tokenizer:
            tokenizer_classes.append(self.rust_tokenizer_class)

        for tokenizer_class in tokenizer_classes:
            tokenizer = tokenizer_class.from_pretrained("albert-base-v2")

            sequences = [
                "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations",
                "ALBERT incorporates two parameter reduction techniques",
                "The first one is a factorized embedding parameterization. By decomposing the large vocabulary embedding matrix into two small matrices, we separate the size of the hidden layers from the size of vocabulary embedding.",  # noqa: E231
            ]

            encoding = tokenizer(sequences, padding=True)
            decoded_sequences = [tokenizer.decode(seq, skip_special_tokens=True) for seq in encoding["input_ids"]]

            # fmt: off
            expected_encoding = {
                'input_ids': [
                    [2, 2953, 45, 21, 13, 10601, 11502, 26, 1119, 8, 8542, 3762, 69, 2477, 16, 816, 18667, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # noqa: E231
                    [2, 2953, 13760, 81, 18906, 5895, 4212, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # noqa: E231
                    [2, 14, 64, 53, 25, 21, 3932, 1333, 11911, 69, 3258, 18906, 1829, 9, 34, 121, 960, 14717, 14, 370, 18630, 11911, 69, 3258, 8187, 77, 81, 284, 24849, 15, 95, 1725, 14, 1072, 16, 14, 3689, 9124, 37, 14, 1072, 16, 18630, 11911, 69, 3258, 9, 3]],  # noqa: E231
                'token_type_ids': [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # noqa: E231
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # noqa: E231
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # noqa: E231
                'attention_mask': [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # noqa: E231
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # noqa: E231
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # noqa: E231
                ]
            }

            expected_decoded_sequence = [
                "albert: a lite bert for self-supervised learning of language representations",
                'albert incorporates two parameter reduction techniques',
                'the first one is a factorized embedding parameterization. by decomposing the large vocabulary embedding matrix into two small matrices, we separate the size of the hidden layers from the size of vocabulary embedding.'  # noqa: E231
            ]
            # fmt: on

            self.assertDictEqual(encoding.data, expected_encoding)

            for expected, decoded in zip(expected_decoded_sequence, decoded_sequences):
                self.assertEqual(expected, decoded)
