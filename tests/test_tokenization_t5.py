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


import unittest

from transformers import SPIECE_UNDERLINE, BatchEncoding, T5Tokenizer, T5TokenizerFast
from transformers.file_utils import cached_property
from transformers.testing_utils import _torch_available, get_tests_dir, require_sentencepiece, require_tokenizers

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")

FRAMEWORK = "pt" if _torch_available else "tf"


@require_sentencepiece
@require_tokenizers
class T5TokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = T5Tokenizer
    rust_tokenizer_class = T5TokenizerFast
    test_rust_tokenizer = True

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

    @cached_property
    def t5_base_tokenizer(self):
        return T5Tokenizer.from_pretrained("t5-base")

    @cached_property
    def t5_base_tokenizer_fast(self):
        return T5TokenizerFast.from_pretrained("t5-base")

    def get_tokenizer(self, **kwargs) -> T5Tokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, pad_token=None, **kwargs)

    def get_rust_tokenizer(self, **kwargs) -> T5TokenizerFast:
        return self.rust_tokenizer_class.from_pretrained(self.tmpdirname, pad_token=None, **kwargs)

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

    def test_eos_treatment(self):
        tokenizer = self.t5_base_tokenizer
        batch_with_eos_added = tokenizer(["hi</s>", "I went to the gym</s>", "</s>"])
        batch_without_eos_added = tokenizer(["hi", "I went to the gym", ""])
        self.assertListEqual(batch_with_eos_added["input_ids"], batch_without_eos_added["input_ids"])

    def test_prepare_seq2seq_batch(self):
        tokenizer = self.t5_base_tokenizer
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        tgt_text = [
            "Summary of the text.",
            "Another summary.",
        ]
        expected_src_tokens = [71, 307, 8986, 21, 4505, 1635, 1707, 5, tokenizer.eos_token_id]
        batch = tokenizer.prepare_seq2seq_batch(
            src_text,
            tgt_texts=tgt_text,
            return_tensors=FRAMEWORK,
        )
        self.assertIsInstance(batch, BatchEncoding)
        result = list(batch.input_ids.numpy()[0])
        self.assertListEqual(expected_src_tokens, result)

        self.assertEqual((2, 9), batch.input_ids.shape)
        self.assertEqual((2, 9), batch.attention_mask.shape)

    def test_empty_target_text(self):
        tokenizer = self.t5_base_tokenizer
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        batch = tokenizer.prepare_seq2seq_batch(src_text, return_tensors=FRAMEWORK)
        # check if input_ids are returned and no decoder_input_ids
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertNotIn("decoder_input_ids", batch)
        self.assertNotIn("decoder_attention_mask", batch)

    def test_max_target_length(self):
        tokenizer = self.t5_base_tokenizer
        src_text = ["A short paragraph for summarization.", "Another short paragraph for summarization."]
        tgt_text = [
            "Summary of the text.",
            "Another summary.",
        ]
        batch = tokenizer.prepare_seq2seq_batch(
            src_text, tgt_texts=tgt_text, max_target_length=32, padding="max_length", return_tensors=FRAMEWORK
        )
        self.assertEqual(32, batch["labels"].shape[1])

        # test None max_target_length
        batch = tokenizer.prepare_seq2seq_batch(
            src_text, tgt_texts=tgt_text, max_length=32, padding="max_length", return_tensors=FRAMEWORK
        )
        self.assertEqual(32, batch["labels"].shape[1])

    def test_outputs_not_longer_than_maxlen(self):
        tokenizer = self.t5_base_tokenizer

        batch = tokenizer.prepare_seq2seq_batch(
            ["I am a small frog" * 1000, "I am a small frog"], return_tensors=FRAMEWORK
        )
        self.assertIsInstance(batch, BatchEncoding)
        self.assertEqual(batch.input_ids.shape, (2, 512))

    def test_eos_in_input(self):
        tokenizer = self.t5_base_tokenizer
        src_text = ["A long paragraph for summarization. </s>"]
        tgt_text = ["Summary of the text. </s>"]
        expected_src_tokens = [71, 307, 8986, 21, 4505, 1635, 1707, 5, 1]
        expected_tgt_tokens = [20698, 13, 8, 1499, 5, 1]

        batch = tokenizer.prepare_seq2seq_batch(src_text, tgt_texts=tgt_text, return_tensors=FRAMEWORK)

        src_ids = list(batch.input_ids.numpy()[0])
        tgt_ids = list(batch.labels.numpy()[0])

        self.assertEqual(expected_src_tokens, src_ids)
        self.assertEqual(expected_tgt_tokens, tgt_ids)
