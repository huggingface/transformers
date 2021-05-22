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

import unittest

from transformers import PegasusTokenizer, PegasusTokenizerFast
from transformers.file_utils import cached_property
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, require_torch

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_no_bos.model")


@require_sentencepiece
@require_tokenizers
class PegasusTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = PegasusTokenizer
    rust_tokenizer_class = PegasusTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = PegasusTokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    @cached_property
    def _large_tokenizer(self):
        return PegasusTokenizer.from_pretrained("google/pegasus-large")

    def get_tokenizer(self, **kwargs) -> PegasusTokenizer:
        return PegasusTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        return ("This is a test", "This is a test")

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<pad>")
        self.assertEqual(vocab_keys[1], "</s>")
        self.assertEqual(vocab_keys[-1], "v")
        self.assertEqual(len(vocab_keys), 1_103)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_103)

    def test_mask_tokens_rust_pegasus(self):
        rust_tokenizer = self.rust_tokenizer_class.from_pretrained(self.tmpdirname)
        py_tokenizer = self.tokenizer_class.from_pretrained(self.tmpdirname)
        raw_input_str = "Let's see which <unk> is the better <unk_token_11> one <mask_1> It seems like this <mask_2> was important </s> <pad> <pad> <pad>"
        rust_ids = rust_tokenizer([raw_input_str], return_tensors=None, add_special_tokens=False).input_ids[0]
        py_ids = py_tokenizer([raw_input_str], return_tensors=None, add_special_tokens=False).input_ids[0]
        self.assertListEqual(py_ids, rust_ids)

    def test_large_mask_tokens(self):
        tokenizer = self._large_tokenizer
        # <mask_1> masks whole sentence while <mask_2> masks single word
        raw_input_str = "<mask_1> To ensure a <mask_2> flow of bank resolutions."
        desired_result = [2, 413, 615, 114, 3, 1971, 113, 1679, 10710, 107, 1]
        ids = tokenizer([raw_input_str], return_tensors=None).input_ids[0]
        self.assertListEqual(desired_result, ids)

    def test_large_tokenizer_settings(self):
        tokenizer = self._large_tokenizer
        # The tracebacks for the following asserts are **better** without messages or self.assertEqual
        assert tokenizer.vocab_size == 96103
        assert tokenizer.pad_token_id == 0
        assert tokenizer.eos_token_id == 1
        assert tokenizer.offset == 103
        assert tokenizer.unk_token_id == tokenizer.offset + 2 == 105
        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.model_max_length == 1024
        raw_input_str = "To ensure a smooth flow of bank resolutions."
        desired_result = [413, 615, 114, 2291, 1971, 113, 1679, 10710, 107, 1]
        ids = tokenizer([raw_input_str], return_tensors=None).input_ids[0]
        self.assertListEqual(desired_result, ids)
        assert tokenizer.convert_ids_to_tokens([0, 1, 2, 3]) == ["<pad>", "</s>", "<mask_1>", "<mask_2>"]

    @require_torch
    def test_large_seq2seq_truncation(self):
        src_texts = ["This is going to be way too long." * 150, "short example"]
        tgt_texts = ["not super long but more than 5 tokens", "tiny"]
        batch = self._large_tokenizer(src_texts, padding=True, truncation=True, return_tensors="pt")
        with self._large_tokenizer.as_target_tokenizer():
            targets = self._large_tokenizer(
                tgt_texts, max_length=5, padding=True, truncation=True, return_tensors="pt"
            )

        assert batch.input_ids.shape == (2, 1024)
        assert batch.attention_mask.shape == (2, 1024)
        assert targets["input_ids"].shape == (2, 5)
        assert len(batch) == 2  # input_ids, attention_mask.


@require_sentencepiece
@require_tokenizers
class BigBirdPegasusTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = PegasusTokenizer
    rust_tokenizer_class = PegasusTokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = PegasusTokenizer(SAMPLE_VOCAB, offset=0, mask_token_sent=None, mask_token="[MASK]")
        tokenizer.save_pretrained(self.tmpdirname)

    @cached_property
    def _large_tokenizer(self):
        return PegasusTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

    def get_tokenizer(self, **kwargs) -> PegasusTokenizer:
        return PegasusTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        return ("This is a test", "This is a test")

    def test_mask_tokens_rust_pegasus(self):
        rust_tokenizer = self.rust_tokenizer_class.from_pretrained(self.tmpdirname)
        py_tokenizer = self.tokenizer_class.from_pretrained(self.tmpdirname)
        raw_input_str = "Let's see which <unk> is the better <unk_token> one [MASK] It seems like this [MASK] was important </s> <pad> <pad> <pad>"
        rust_ids = rust_tokenizer([raw_input_str], return_tensors=None, add_special_tokens=False).input_ids[0]
        py_ids = py_tokenizer([raw_input_str], return_tensors=None, add_special_tokens=False).input_ids[0]
        self.assertListEqual(py_ids, rust_ids)

    @require_torch
    def test_large_seq2seq_truncation(self):
        src_texts = ["This is going to be way too long." * 1000, "short example"]
        tgt_texts = ["not super long but more than 5 tokens", "tiny"]
        batch = self._large_tokenizer(src_texts, padding=True, truncation=True, return_tensors="pt")
        with self._large_tokenizer.as_target_tokenizer():
            targets = self._large_tokenizer(
                tgt_texts, max_length=5, padding=True, truncation=True, return_tensors="pt"
            )

        assert batch.input_ids.shape == (2, 4096)
        assert batch.attention_mask.shape == (2, 4096)
        assert targets["input_ids"].shape == (2, 5)
        assert len(batch) == 2  # input_ids, attention_mask.

    def test_equivalence_to_orig_tokenizer(self):
        """
        To run with original TF tokenizer:

        !wget https://github.com/google-research/bigbird/raw/master/bigbird/vocab/pegasus.model
        !pip install tensorflow-text

        import tensorflow.compat.v2 as tf
        import tensorflow_text as tft

        VOCAB_FILE = "./pegasus.model"

        tf.enable_v2_behavior()

        test_str = "This is an example string that is used to test the original TF implementation against the HF implementation"
        tokenizer = tft.SentencepieceTokenizer(model=tf.io.gfile.GFile(VOCAB_FILE, "rb").read())

        tokenizer.tokenize(test_str)
        """

        test_str = "This is an example string that is used to test the original TF implementation against the HF implementation"

        token_ids = self._large_tokenizer(test_str).input_ids

        self.assertListEqual(
            token_ids,
            [182, 117, 142, 587, 4211, 120, 117, 263, 112, 804, 109, 856, 25016, 3137, 464, 109, 26955, 3137, 1],
        )
