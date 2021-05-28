# coding=utf-8
# Copyright 2018 HuggingFace Inc. team.
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

from transformers import CamembertTokenizer, CamembertTokenizerFast
from transformers.file_utils import is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, slow

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece.model")
SAMPLE_BPE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece_bpe.model")

FRAMEWORK = "pt" if is_torch_available() else "tf"


@require_sentencepiece
@require_tokenizers
class CamembertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = CamembertTokenizer
    rust_tokenizer_class = CamembertTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = CamembertTokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<pad>"
        token_id = 1

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<s>NOTUSED")
        self.assertEqual(vocab_keys[1], "<pad>")
        self.assertEqual(vocab_keys[-1], "<mask>")
        self.assertEqual(len(vocab_keys), 1_004)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_005)

    def test_rust_and_python_bpe_tokenizers(self):
        tokenizer = CamembertTokenizer(SAMPLE_BPE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)
        rust_tokenizer = CamembertTokenizerFast.from_pretrained(self.tmpdirname)

        sequence = "I was born in 92000, and this is falsé."

        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        # <unk> tokens are not the same for `rust` than for `slow`.
        # Because spm gives back raw token instead of `unk` in EncodeAsPieces
        # tokens = tokenizer.tokenize(sequence)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)

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
        expected_encoding = {'input_ids': [[5, 27879, 10, 38, 3993, 81, 1107, 28992, 255, 33, 10, 387, 105, 2505, 751, 26, 8577, 5281, 2242, 1168, 387, 105, 2505, 751, 26, 7881, 15848, 816, 26, 2872, 53, 909, 15618, 10, 23052, 26, 16513, 15767, 7627, 10, 38, 20703, 7, 361, 9608, 5428, 7, 3425, 20703, 55, 7, 16093, 465, 7, 4194, 110, 62, 385, 3867, 7, 16093, 22285, 1276, 1782, 28854, 61, 14112, 491, 24613, 25608, 402, 38, 607, 15165, 53, 1168, 28854, 61, 14112, 491, 10452, 5250, 38, 607, 27308, 53, 1466, 14976, 2527, 1754, 3591, 15848, 816, 16320, 10, 378, 779, 1754, 13, 14112, 3169, 1168, 8, 3737, 1361, 15940, 31859, 21, 11045, 4046, 90, 121, 4653, 7, 275, 105, 412, 451, 751, 1168, 8320, 10, 451, 449, 6685, 9, 6], [5, 21, 20703, 2856, 1160, 816, 1200, 3591, 26, 15848, 8, 3737, 2008, 18574, 341, 28375, 2770, 7142, 23, 321, 3280, 816, 21, 12584, 1938, 6741, 1107, 1853, 402, 91, 15003, 133, 16, 5285, 1168, 21, 12191, 1113, 12584, 378, 6051, 13, 6239, 10, 9, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [5, 908, 31, 1806, 5202, 12363, 9557, 290, 76, 12294, 10, 14976, 808, 13, 7058, 18, 4251, 9, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E501
        # fmt: on

        # camembert is a french model. So we also use french texts.
        sequences = [
            "Le transformeur est un modèle d'apprentissage profond introduit en 2017, "
            "utilisé principalement dans le domaine du traitement automatique des langues (TAL).",
            "À l'instar des réseaux de neurones récurrents (RNN), les transformeurs sont conçus "
            "pour gérer des données séquentielles, telles que le langage naturel, pour des tâches "
            "telles que la traduction et la synthèse de texte.",
        ]

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="camembert-base",
            revision="3a0641d9a1aeb7e848a74299e7e4c4bca216b4cf",
        )
