import json
import os
import unittest

from transformers import BartTokenizer, BartTokenizerFast, BatchEncoding
from transformers.file_utils import cached_property
from transformers.testing_utils import require_torch
from transformers.tokenization_roberta import VOCAB_FILES_NAMES

from .test_tokenization_common import TokenizerTesterMixin


class TestTokenizationBart(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = BartTokenizer

    def setUp(self):
        super().setUp()
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return BartTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        return "lower newer", "lower newer"

    @cached_property
    def default_tokenizer(self):
        return BartTokenizer.from_pretrained("facebook/bart-large")

    @cached_property
    def default_tokenizer_fast(self):
        return BartTokenizerFast.from_pretrained("facebook/bart-large")

    @require_torch
    def test_prepare_seq2seq_batch(self):
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        tgt_text = [
            "Summary of the text.",
            "Another summary.",
        ]
        expected_src_tokens = [0, 250, 251, 17818, 13, 39186, 1938, 4, 2]

        for tokenizer in [self.default_tokenizer, self.default_tokenizer_fast]:
            batch = tokenizer.prepare_seq2seq_batch(
                src_text, tgt_texts=tgt_text, max_length=len(expected_src_tokens), return_tensors="pt"
            )
            self.assertIsInstance(batch, BatchEncoding)

            self.assertEqual((2, 9), batch.input_ids.shape)
            self.assertEqual((2, 9), batch.attention_mask.shape)
            result = batch.input_ids.tolist()[0]
            self.assertListEqual(expected_src_tokens, result)
            # Test that special tokens are reset

    # Test Prepare Seq
    @require_torch
    def test_seq2seq_batch_empty_target_text(self):
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        for tokenizer in [self.default_tokenizer, self.default_tokenizer_fast]:
            batch = tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt")
            # check if input_ids are returned and no labels
            self.assertIn("input_ids", batch)
            self.assertIn("attention_mask", batch)
            self.assertNotIn("labels", batch)
            self.assertNotIn("decoder_attention_mask", batch)

    @require_torch
    def test_seq2seq_batch_max_target_length(self):
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        tgt_text = [
            "Summary of the text.",
            "Another summary.",
        ]
        for tokenizer in [self.default_tokenizer, self.default_tokenizer_fast]:
            batch = tokenizer.prepare_seq2seq_batch(
                src_text, tgt_texts=tgt_text, max_target_length=32, padding="max_length", return_tensors="pt"
            )
            self.assertEqual(32, batch["labels"].shape[1])

            # test None max_target_length
            batch = tokenizer.prepare_seq2seq_batch(
                src_text, tgt_texts=tgt_text, max_length=32, padding="max_length", return_tensors="pt"
            )
            self.assertEqual(32, batch["labels"].shape[1])

    @require_torch
    def test_seq2seq_batch_not_longer_than_maxlen(self):
        for tokenizer in [self.default_tokenizer, self.default_tokenizer_fast]:
            batch = tokenizer.prepare_seq2seq_batch(
                ["I am a small frog" * 1024, "I am a small frog"], return_tensors="pt"
            )
            self.assertIsInstance(batch, BatchEncoding)
            self.assertEqual(batch.input_ids.shape, (2, 1024))

    @require_torch
    def test_special_tokens(self):

        src_text = ["A long paragraph for summarization."]
        tgt_text = [
            "Summary of the text.",
        ]
        for tokenizer in [self.default_tokenizer, self.default_tokenizer_fast]:
            batch = tokenizer.prepare_seq2seq_batch(src_text, tgt_texts=tgt_text, return_tensors="pt")
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            self.assertTrue((input_ids[:, 0] == tokenizer.bos_token_id).all().item())
            self.assertTrue((labels[:, 0] == tokenizer.bos_token_id).all().item())
            self.assertTrue((input_ids[:, -1] == tokenizer.eos_token_id).all().item())
            self.assertTrue((labels[:, -1] == tokenizer.eos_token_id).all().item())
