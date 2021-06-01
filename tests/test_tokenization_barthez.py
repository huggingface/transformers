# coding=utf-8
# Copyright 2020 Ecole Polytechnique and HuggingFace Inc. team.
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

from transformers import BarthezTokenizer, BarthezTokenizerFast, BatchEncoding
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow

from .test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
@require_sentencepiece
@slow  # see https://github.com/huggingface/transformers/issues/11457
class BarthezTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BarthezTokenizer
    rust_tokenizer_class = BarthezTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        tokenizer = BarthezTokenizerFast.from_pretrained("moussaKam/mbarthez")
        tokenizer.save_pretrained(self.tmpdirname)
        tokenizer.save_pretrained(self.tmpdirname, legacy_format=False)
        self.tokenizer = tokenizer

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<pad>"
        token_id = 1

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<s>")
        self.assertEqual(vocab_keys[1], "<pad>")
        self.assertEqual(vocab_keys[-1], "<mask>")
        self.assertEqual(len(vocab_keys), 101_122)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 101_122)

    @require_torch
    def test_prepare_batch(self):
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        expected_src_tokens = [0, 57, 3018, 70307, 91, 2]

        batch = self.tokenizer(
            src_text, max_length=len(expected_src_tokens), padding=True, truncation=True, return_tensors="pt"
        )
        self.assertIsInstance(batch, BatchEncoding)

        self.assertEqual((2, 6), batch.input_ids.shape)
        self.assertEqual((2, 6), batch.attention_mask.shape)
        result = batch.input_ids.tolist()[0]
        self.assertListEqual(expected_src_tokens, result)

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

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[0, 490, 14328, 4507, 354, 47, 43669, 95, 25, 78117, 20215, 19779, 190, 22, 400, 4, 35343, 80310, 603, 86, 24937, 105, 33438, 94762, 196, 39642, 7, 15, 15933, 173, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 10534, 87, 25, 66, 3358, 196, 55289, 8, 82961, 81, 2204, 75203, 7, 15, 763, 12956, 216, 178, 14328, 9595, 1377, 69693, 7, 448, 71021, 196, 18106, 1437, 13974, 108, 9083, 4, 49315, 7, 39, 86, 1326, 2793, 46333, 4, 448, 196, 74588, 7, 49315, 7, 39, 21, 822, 38470, 74, 21, 66723, 62480, 8, 22050, 5, 2]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # noqa: E501
        # fmt: on

        # moussaKam/mbarthez is a french model. So we also use french texts.
        sequences = [
            "Le transformeur est un modèle d'apprentissage profond introduit en 2017, "
            "utilisé principalement dans le domaine du traitement automatique des langues (TAL).",
            "À l'instar des réseaux de neurones récurrents (RNN), les transformeurs sont conçus "
            "pour gérer des données séquentielles, telles que le langage naturel, pour des tâches "
            "telles que la traduction et la synthèse de texte.",
        ]

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="moussaKam/mbarthez",
            revision="c2e4ecbca5e3cd2c37fe1ac285ca4fbdf1366fb6",
            sequences=sequences,
        )
