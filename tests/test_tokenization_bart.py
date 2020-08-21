import unittest

from transformers import BartTokenizer, BartTokenizerFast, BatchEncoding
from transformers.file_utils import cached_property


class TestTokenizationBart(unittest.TestCase):
    @cached_property
    def default_tokenizer(self):
        return BartTokenizer.from_pretrained("facebook/bart-large")

    @cached_property
    def default_tokenizer_fast(self):
        return BartTokenizerFast.from_pretrained("facebook/bart-large")

    def test_prepare_seq2seq_batch(self):
        tokenizers = [self.default_tokenizer, self.default_tokenizer_fast]
        src_text = ["A long paragraph for summrization.", "Another paragraph for summrization."]
        tgt_text = [
            "Summary of the text.",
            "Another summary.",
        ]
        expected_src_tokens = [0, 250, 251, 17818, 13, 32933, 21645, 1258, 4, 2]

        for tokenizer in tokenizers:
            batch = tokenizer.prepare_seq2seq_batch(
                src_text, tgt_texts=tgt_text, max_length=len(expected_src_tokens), return_tensors="pt"
            )
            self.assertIsInstance(batch, BatchEncoding)

            self.assertEqual((2, 10), batch.input_ids.shape)
            self.assertEqual((2, 10), batch.attention_mask.shape)
            result = batch.input_ids.tolist()[0]
            self.assertListEqual(expected_src_tokens, result)
            # Test that special tokens are reset

    def test_empty_target_text(self):
        tokenizers = [self.default_tokenizer, self.default_tokenizer_fast]
        src_text = ["A long paragraph for summrization.", "Another paragraph for summrization."]
        for tokenizer in tokenizers:
            batch = tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt")
            # check if input_ids are returned and no labels
            self.assertIn("input_ids", batch)
            self.assertIn("attention_mask", batch)
            self.assertNotIn("labels", batch)
            self.assertNotIn("decoder_attention_mask", batch)

    def test_max_target_length(self):
        tokenizers = [self.default_tokenizer, self.default_tokenizer_fast]
        src_text = ["A long paragraph for summrization.", "Another paragraph for summrization."]
        tgt_text = [
            "Summary of the text.",
            "Another summary.",
        ]
        for tokenizer in tokenizers:
            batch = tokenizer.prepare_seq2seq_batch(
                src_text, tgt_texts=tgt_text, max_target_length=32, padding="max_length", return_tensors="pt"
            )
            self.assertEqual(32, batch["labels"].shape[1])

            # test None max_target_length
            batch = tokenizer.prepare_seq2seq_batch(
                src_text, tgt_texts=tgt_text, max_length=32, padding="max_length", return_tensors="pt"
            )
            self.assertEqual(32, batch["labels"].shape[1])

    def test_outputs_not_longer_than_maxlen(self):
        tokenizers = [self.default_tokenizer, self.default_tokenizer_fast]

        for tokenizer in tokenizers:
            batch = tokenizer.prepare_seq2seq_batch(
                ["I am a small frog" * 1024, "I am a small frog"], return_tensors="pt"
            )
            self.assertIsInstance(batch, BatchEncoding)
            self.assertEqual(batch.input_ids.shape, (2, 1024))

    def test_special_tokens(self):
        tokenizers = [self.default_tokenizer, self.default_tokenizer_fast]
        src_text = ["A long paragraph for summrization."]
        tgt_text = [
            "Summary of the text.",
        ]
        for tokenizer in tokenizers:
            batch = tokenizer.prepare_seq2seq_batch(src_text, tgt_texts=tgt_text, return_tensors="pt")
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            self.assertTrue((input_ids[:, 0] == tokenizer.bos_token_id).all().item())
            self.assertTrue((labels[:, 0] == tokenizer.bos_token_id).all().item())
            self.assertTrue((input_ids[:, -1] == tokenizer.eos_token_id).all().item())
            self.assertTrue((labels[:, -1] == tokenizer.eos_token_id).all().item())
