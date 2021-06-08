# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import BigBirdTokenizer, BigBirdTokenizerFast
from transformers.file_utils import cached_property
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow

from .test_tokenization_common import TokenizerTesterMixin


SPIECE_UNDERLINE = "▁"

SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class BigBirdTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BigBirdTokenizer
    rust_tokenizer_class = BigBirdTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        tokenizer = self.tokenizer_class(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<s>"
        token_id = 1

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<unk>")
        self.assertEqual(vocab_keys[1], "<s>")
        self.assertEqual(vocab_keys[-1], "[MASK]")
        self.assertEqual(len(vocab_keys), 1_004)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_000)

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
        tokenizer = BigBirdTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [285, 46, 10, 170, 382],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "é",
                ".",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 0, 4],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "<unk>",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "<unk>",
                ".",
            ],
        )

    @cached_property
    def big_tokenizer(self):
        return BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

    @slow
    def test_tokenization_base_easy_symbols(self):
        symbols = "Hello World!"
        original_tokenizer_encodings = [65, 18536, 2260, 101, 66]

        self.assertListEqual(original_tokenizer_encodings, self.big_tokenizer.encode(symbols))

    @slow
    def test_tokenization_base_hard_symbols(self):
        symbols = 'This is a very long text with a lot of weird characters, such as: . , ~ ? ( ) " [ ] ! : - . Also we will add words that should not exsist and be tokenized to <unk>, such as saoneuhaoesuth'
        # fmt: off
        original_tokenizer_encodings = [65, 871, 419, 358, 946, 991, 2521, 452, 358, 1357, 387, 7751, 3536, 112, 985, 456, 126, 865, 938, 5400, 5734, 458, 1368, 467, 786, 2462, 5246, 1159, 633, 865, 4519, 457, 582, 852, 2557, 427, 916, 508, 405, 34324, 497, 391, 408, 11342, 1244, 385, 100, 938, 985, 456, 574, 362, 12597, 3200, 3129, 1172, 66]  # noqa: E231
        # fmt: on
        self.assertListEqual(original_tokenizer_encodings, self.big_tokenizer.encode(symbols))

    @require_torch
    @slow
    def test_torch_encode_plus_sent_to_model(self):
        import torch

        from transformers import BigBirdConfig, BigBirdModel

        # Build sequence
        first_ten_tokens = list(self.big_tokenizer.get_vocab().keys())[:10]
        sequence = " ".join(first_ten_tokens)
        encoded_sequence = self.big_tokenizer.encode_plus(sequence, return_tensors="pt", return_token_type_ids=False)
        batch_encoded_sequence = self.big_tokenizer.batch_encode_plus(
            [sequence + " " + sequence], return_tensors="pt", return_token_type_ids=False
        )

        config = BigBirdConfig(attention_type="original_full")
        model = BigBirdModel(config)

        assert model.get_input_embeddings().weight.shape[0] >= self.big_tokenizer.vocab_size

        with torch.no_grad():
            model(**encoded_sequence)
            model(**batch_encoded_sequence)

    @slow
    def test_special_tokens(self):
        """
        To reproduce:

        $ wget https://github.com/google-research/bigbird/blob/master/bigbird/vocab/gpt2.model?raw=true
        $ mv gpt2.model?raw=true gpt2.model

        ```
        import tensorflow_text as tft
        import tensorflow as tf

        vocab_model_file = "./gpt2.model"
        tokenizer = tft.SentencepieceTokenizer(model=tf.io.gfile.GFile(vocab_model_file, "rb").read()))
        ids = tokenizer.tokenize("Paris is the [MASK].")
        ids = tf.concat([tf.constant([65]), ids, tf.constant([66])], axis=0)
        detokenized = tokenizer.detokenize(ids)  # should give [CLS] Paris is the [MASK].[SEP]
        """
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        decoded_text = tokenizer.decode(tokenizer("Paris is the [MASK].").input_ids)

        self.assertTrue(decoded_text == "[CLS] Paris is the [MASK].[SEP]")

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[65, 39286, 458, 36335, 2001, 456, 13073, 13266, 455, 113, 7746, 1741, 11157, 391, 13073, 13266, 455, 113, 3967, 35412, 113, 4936, 109, 3870, 2377, 113, 30084, 45720, 458, 134, 17496, 112, 503, 11672, 113, 118, 112, 5665, 13347, 38687, 112, 1496, 31389, 112, 3268, 47264, 134, 962, 112, 16377, 8035, 23130, 430, 12169, 15518, 28592, 458, 146, 41697, 109, 391, 12169, 15518, 16689, 458, 146, 41358, 109, 452, 726, 4034, 111, 763, 35412, 5082, 388, 1903, 111, 9051, 391, 2870, 48918, 1900, 1123, 550, 998, 112, 9586, 15985, 455, 391, 410, 22955, 37636, 114, 66], [65, 448, 17496, 419, 3663, 385, 763, 113, 27533, 2870, 3283, 13043, 1639, 24713, 523, 656, 24013, 18550, 2521, 517, 27014, 21244, 420, 1212, 1465, 391, 927, 4833, 388, 578, 11786, 114, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [65, 484, 2169, 7687, 21932, 18146, 726, 363, 17032, 3391, 114, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E501
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="google/bigbird-roberta-base",
            revision="215c99f1600e06f83acce68422f2035b2b5c3510",
        )
