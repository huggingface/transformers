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

import tempfile
import unittest

from transformers import AddedToken, CamembertTokenizer, CamembertTokenizerFast
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, slow
from transformers.utils import is_torch_available

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")
SAMPLE_BPE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_bpe.model")

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

    @unittest.skip(
        "Token maps are not equal because someone set the probability of ('<unk>NOTUSED', -100), so it's never encoded for fast"
    )
    def test_special_tokens_map_equal(self):
        return

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<pad>"
        token_id = 1  # 1 is the offset id, but in the spm vocab it's 3

        self.assertEqual(self.get_tokenizer().convert_tokens_to_ids(token), token_id)
        self.assertEqual(self.get_tokenizer().convert_ids_to_tokens(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<s>NOTUSED")
        self.assertEqual(vocab_keys[1], "<pad>")
        self.assertEqual(vocab_keys[-1], "<mask>")
        self.assertEqual(len(vocab_keys), 1_005)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_000)

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
        expected_encoding = {'input_ids': [[5, 54, 7196, 297, 30, 23, 776, 18, 11, 3215, 3705, 8252, 22, 3164, 1181, 2116, 29, 16, 813, 25, 791, 3314, 20, 3446, 38, 27575, 120, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [5, 468, 17, 11, 9088, 20, 1517, 8, 22804, 18818, 10, 38, 629, 607, 607, 142, 19, 7196, 867, 56, 10326, 24, 2267, 20, 416, 5072, 15612, 233, 734, 7, 2399, 27, 16, 3015, 1649, 7, 24, 20, 4338, 2399, 27, 13, 3400, 14, 13, 6189, 8, 930, 9, 6]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # noqa: E501
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
            sequences=sequences,
        )

    # Overwritten because we have to use from slow (online pretrained is wrong, the tokenizer.json has a whole)
    def test_added_tokens_serialization(self):
        self.maxDiff = None

        # Utility to test the added vocab
        def _test_added_vocab_and_eos(expected, tokenizer_class, expected_eos, temp_dir):
            tokenizer = tokenizer_class.from_pretrained(temp_dir)
            self.assertTrue(str(expected_eos) not in tokenizer.additional_special_tokens)
            self.assertIn(new_eos, tokenizer.added_tokens_decoder.values())
            self.assertEqual(tokenizer.added_tokens_decoder[tokenizer.eos_token_id], new_eos)
            self.assertDictEqual(expected, tokenizer.added_tokens_decoder)
            return tokenizer

        new_eos = AddedToken("[NEW_EOS]", rstrip=False, lstrip=True, normalized=False)
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                # Load a slow tokenizer from the hub, init with the new token for fast to also include it
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, eos_token=new_eos)
                EXPECTED_ADDED_TOKENS_DECODER = tokenizer.added_tokens_decoder
                with self.subTest("Hub -> Slow: Test loading a slow tokenizer from the hub)"):
                    self.assertEqual(tokenizer._eos_token, new_eos)
                    self.assertIn(new_eos, list(tokenizer.added_tokens_decoder.values()))

                with tempfile.TemporaryDirectory() as tmp_dir_2:
                    tokenizer.save_pretrained(tmp_dir_2)
                    with self.subTest(
                        "Hub -> Slow -> Slow: Test saving this slow tokenizer and reloading it in the fast class"
                    ):
                        _test_added_vocab_and_eos(
                            EXPECTED_ADDED_TOKENS_DECODER, self.tokenizer_class, new_eos, tmp_dir_2
                        )

                    if self.rust_tokenizer_class is not None:
                        with self.subTest(
                            "Hub -> Slow -> Fast: Test saving this slow tokenizer and reloading it in the fast class"
                        ):
                            tokenizer_fast = _test_added_vocab_and_eos(
                                EXPECTED_ADDED_TOKENS_DECODER, self.rust_tokenizer_class, new_eos, tmp_dir_2
                            )
                            with tempfile.TemporaryDirectory() as tmp_dir_3:
                                tokenizer_fast.save_pretrained(tmp_dir_3)
                                with self.subTest(
                                    "Hub -> Slow -> Fast -> Fast: Test saving this fast tokenizer and reloading it in the fast class"
                                ):
                                    _test_added_vocab_and_eos(
                                        EXPECTED_ADDED_TOKENS_DECODER, self.rust_tokenizer_class, new_eos, tmp_dir_3
                                    )

                                with self.subTest(
                                    "Hub -> Slow -> Fast -> Slow: Test saving this slow tokenizer and reloading it in the slow class"
                                ):
                                    _test_added_vocab_and_eos(
                                        EXPECTED_ADDED_TOKENS_DECODER, self.rust_tokenizer_class, new_eos, tmp_dir_3
                                    )

                with self.subTest("Hub -> Fast: Test loading a fast tokenizer from the hub)"):
                    if self.rust_tokenizer_class is not None:
                        tokenizer_fast = self.rust_tokenizer_class.from_pretrained(
                            pretrained_name, eos_token=new_eos, from_slow=True
                        )
                        self.assertEqual(tokenizer_fast._eos_token, new_eos)
                        self.assertIn(new_eos, list(tokenizer_fast.added_tokens_decoder.values()))
                        # We can't test the following because for BC we kept the default rstrip lstrip in slow not fast. Will comment once normalization is alright
                        with self.subTest("Hub -> Fast == Hub -> Slow: make sure slow and fast tokenizer match"):
                            self.assertDictEqual(EXPECTED_ADDED_TOKENS_DECODER, tokenizer_fast.added_tokens_decoder)

                        EXPECTED_ADDED_TOKENS_DECODER = tokenizer_fast.added_tokens_decoder
                        with tempfile.TemporaryDirectory() as tmp_dir_4:
                            tokenizer_fast.save_pretrained(tmp_dir_4)
                            with self.subTest("Hub -> Fast -> Fast: saving Fast1 locally and loading"):
                                _test_added_vocab_and_eos(
                                    EXPECTED_ADDED_TOKENS_DECODER, self.rust_tokenizer_class, new_eos, tmp_dir_4
                                )

                            with self.subTest("Hub -> Fast -> Slow: saving Fast1 locally and loading"):
                                _test_added_vocab_and_eos(
                                    EXPECTED_ADDED_TOKENS_DECODER, self.tokenizer_class, new_eos, tmp_dir_4
                                )
