# coding=utf-8
# Copyright 2023 The HuggingFace Inc. and Baidu team. All rights reserved.
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
""" Testing suite for the PyTorch ErnieM model. """

import unittest

from transformers import ErnieMTokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/spiece.model")


@require_sentencepiece
@require_tokenizers
class ErnieMTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = ErnieMTokenizer
    test_seq2seq = False
    test_sentencepiece = True
    test_rust_tokenizer = False
    test_sentencepiece_ignore_case = False

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = ErnieMTokenizer(SAMPLE_VOCAB, unk_token="<unk>", pad_token="<pad>")
        tokenizer.save_pretrained(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<pad>"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<pad>")
        self.assertEqual(vocab_keys[1], "<unk>")
        self.assertEqual(vocab_keys[-1], "▁eloquent")
        self.assertEqual(len(vocab_keys), 30_000)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 30_000)

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
        tokenizer = ErnieMTokenizer(SAMPLE_VOCAB, do_lower_case=True, unk_token="<unk>", pad_token="<pad>")

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁this", "▁is", "▁a", "▁test"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [48, 25, 21, 1289])

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        # ErnieMTokenizer(paddlenlp implementation) outputs '9' instead of '_9' so to mimic that '_9' is changed to '9'
        self.assertListEqual(
            tokens, ["▁i", "▁was", "▁born", "▁in", "9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "é", "."]
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [31, 23, 386, 19, 518, 3050, 15, 17, 48, 25, 8256, 18, 1, 9])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            ["▁i", "▁was", "▁born", "▁in", "9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "."],
        )

    def test_sequence_builders(self):
        tokenizer = ErnieMTokenizer(SAMPLE_VOCAB, unk_token="<unk>", pad_token="<pad>")

        text = tokenizer.encode("sequence builders")
        text_2 = tokenizer.encode("multi-sequence build")

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + [
            tokenizer.sep_token_id
        ] + text_2 + [tokenizer.sep_token_id]

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[0, 11062, 82772, 7, 15, 82772, 538, 51529, 237, 17198, 1290, 206, 9, 215175, 1314, 136, 17198, 1290, 206, 9, 56359, 42, 122009, 9, 16466, 16, 87344, 4537, 9, 4717, 78381, 6, 159958, 7, 15, 24480, 618, 4, 527, 22693, 9, 304, 4, 2777, 24480, 9874, 4, 43523, 594, 4, 803, 18392, 33189, 18, 4, 43523, 24447, 5, 5, 5, 16, 100, 24955, 83658, 9626, 144057, 15, 839, 22335, 16, 136, 24955, 83658, 83479, 15, 39102, 724, 16, 678, 645, 6460, 1328, 4589, 42, 122009, 115774, 23, 3559, 1328, 46876, 7, 136, 53894, 1940, 42227, 41159, 17721, 823, 425, 4, 27512, 98722, 206, 136, 5531, 4970, 919, 17336, 5, 2], [0, 20080, 618, 83, 82775, 47, 479, 9, 1517, 73, 53894, 333, 80581, 110117, 18811, 5256, 1295, 51, 152526, 297, 7986, 390, 124416, 538, 35431, 214, 98, 15044, 25737, 136, 7108, 43701, 23, 756, 135355, 7, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 581, 63773, 119455, 6, 147797, 88203, 7, 645, 70, 21, 3285, 10269, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="susnato/ernie-m-base_pytorch",
            sequences=[
                "Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides "
                "general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet...) for Natural "
                "Language Understanding (NLU) and Natural Language Generation (NLG) with over32+ pretrained "
                "models in100+ languages and deep interoperability between Jax, PyTorch and TensorFlow.",
                "BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly "
                "conditioning on both left and right context in all layers.",
                "The quick brown fox jumps over the lazy dog.",
            ],
        )
