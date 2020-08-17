# coding=utf-8
# Copyright 2018 Google T5 Authors and HuggingFace Inc. team.
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

from transformers import BatchEncoding
from transformers.testing_utils import _torch_available
from transformers.tokenization_t5 import T5Tokenizer
from transformers.tokenization_xlnet import SPIECE_UNDERLINE

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece.model")

FRAMEWORK = "pt" if _torch_available else "tf"


class T5TokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = T5Tokenizer

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = T5Tokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_full_tokenizer(self):
        tokenizer = T5Tokenizer(SAMPLE_VOCAB)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [285, 46, 10, 170, 382])

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
        self.assertListEqual(ids, [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 0, 4])

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

    def test_prepare_seq2seq_batch(self):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        src_text = ["A long paragraph for summrization.", "Another paragraph for summrization."]
        tgt_text = [
            "Summary of the text.",
            "Another summary.",
        ]
        expected_src_tokens = [71, 307, 8986, 21, 4505, 51, 52, 1707, 5]
        batch = tokenizer.prepare_seq2seq_batch(
            src_text, tgt_texts=tgt_text, max_length=len(expected_src_tokens), return_tensors=FRAMEWORK
        )
        self.assertIsInstance(batch, BatchEncoding)

        self.assertEqual((2, 9), batch.input_ids.shape)
        self.assertEqual((2, 9), batch.attention_mask.shape)
        result = list(batch.input_ids.numpy()[0])
        self.assertListEqual(expected_src_tokens, result)
        # Test that special tokens are reset
        self.assertEqual(tokenizer.prefix_tokens, [])

    def test_empty_target_text(self):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        src_text = ["A long paragraph for summrization.", "Another paragraph for summrization."]
        batch = tokenizer.prepare_seq2seq_batch(src_text, return_tensors=FRAMEWORK)
        # check if input_ids are returned and no decoder_input_ids
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertNotIn("decoder_input_ids", batch)
        self.assertNotIn("decoder_attention_mask", batch)

    def test_max_target_length(self):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        src_text = ["A long paragraph for summrization.", "Another paragraph for summrization."]
        tgt_text = [
            "Summary of the text.",
            "Another summary.",
        ]
        batch = tokenizer.prepare_seq2seq_batch(
            src_text, tgt_texts=tgt_text, max_target_length=32, padding="max_length", return_tensors=FRAMEWORK
        )
        self.assertEqual(32, batch["decoder_input_ids"].shape[1])
        self.assertEqual(32, batch["decoder_attention_mask"].shape[1])

        # test None max_target_length
        batch = tokenizer.prepare_seq2seq_batch(
            src_text, tgt_texts=tgt_text, max_length=32, padding="max_length", return_tensors=FRAMEWORK
        )
        self.assertEqual(32, batch["decoder_input_ids"].shape[1])
        self.assertEqual(32, batch["decoder_attention_mask"].shape[1])

    def test_outputs_not_longer_than_maxlen(self):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")

        batch = tokenizer.prepare_seq2seq_batch(
            ["I am a small frog" * 1000, "I am a small frog"], return_tensors=FRAMEWORK
        )
        self.assertIsInstance(batch, BatchEncoding)
        self.assertEqual(batch.input_ids.shape, (2, 512))

    def test_eos_in_input(self):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        src_text = ["A long paragraph for summrization. </s>"]
        tgt_text = ["Summary of the text. </s>"]
        expected_src_tokens = [71, 307, 8986, 21, 4505, 51, 52, 1707, 5, 1]
        expected_tgt_tokens = [0, 20698, 13, 8, 1499, 5, 1]

        batch = tokenizer.prepare_seq2seq_batch(src_text, tgt_texts=tgt_text, return_tensors=FRAMEWORK)

        src_ids = list(batch.input_ids.numpy()[0])
        tgt_ids = list(batch.decoder_input_ids.numpy()[0])

        self.assertEqual(expected_src_tokens, src_ids)
        self.assertEqual(expected_tgt_tokens, tgt_ids)
