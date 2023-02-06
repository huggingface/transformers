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
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, require_torch, slow
from transformers.utils import cached_property

from ...test_tokenization_common import TokenizerTesterMixin


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

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "</s>"
        token_id = 1

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

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
        raw_input_str = (
            "Let's see which <unk> is the better <unk_token_11> one <mask_1> It seems like this <mask_2> was important"
            " </s> <pad> <pad> <pad>"
        )
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
        targets = self._large_tokenizer(
            text_target=tgt_texts, max_length=5, padding=True, truncation=True, return_tensors="pt"
        )

        assert batch.input_ids.shape == (2, 1024)
        assert batch.attention_mask.shape == (2, 1024)
        assert targets["input_ids"].shape == (2, 5)
        assert len(batch) == 2  # input_ids, attention_mask.

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[38979, 143, 18485, 606, 130, 26669, 87686, 121, 54189, 1129, 111, 26669, 87686, 121, 9114, 14787, 121, 13249, 158, 592, 956, 121, 14621, 31576, 143, 62613, 108, 9688, 930, 43430, 11562, 62613, 304, 108, 11443, 897, 108, 9314, 17415, 63399, 108, 11443, 7614, 18316, 118, 4284, 7148, 12430, 143, 1400, 25703, 158, 111, 4284, 7148, 11772, 143, 21297, 1064, 158, 122, 204, 3506, 1754, 1133, 14787, 1581, 115, 33224, 4482, 111, 1355, 110, 29173, 317, 50833, 108, 20147, 94665, 111, 77198, 107, 1], [110, 62613, 117, 638, 112, 1133, 121, 20098, 1355, 79050, 13872, 135, 1596, 53541, 1352, 141, 13039, 5542, 124, 302, 518, 111, 268, 2956, 115, 149, 4427, 107, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [139, 1235, 2799, 18289, 17780, 204, 109, 9474, 1296, 107, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E501
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="google/bigbird-pegasus-large-arxiv",
            revision="ba85d0851d708441f91440d509690f1ab6353415",
        )


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
        raw_input_str = (
            "Let's see which <unk> is the better <unk_token> one [MASK] It seems like this [MASK] was important </s>"
            " <pad> <pad> <pad>"
        )
        rust_ids = rust_tokenizer([raw_input_str], return_tensors=None, add_special_tokens=False).input_ids[0]
        py_ids = py_tokenizer([raw_input_str], return_tensors=None, add_special_tokens=False).input_ids[0]
        self.assertListEqual(py_ids, rust_ids)

    @require_torch
    def test_large_seq2seq_truncation(self):
        src_texts = ["This is going to be way too long." * 1000, "short example"]
        tgt_texts = ["not super long but more than 5 tokens", "tiny"]
        batch = self._large_tokenizer(src_texts, padding=True, truncation=True, return_tensors="pt")
        targets = self._large_tokenizer(
            text_target=tgt_texts, max_length=5, padding=True, truncation=True, return_tensors="pt"
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

        test_str = (
            "This is an example string that is used to test the original TF implementation against the HF"
            " implementation"
        )

        token_ids = self._large_tokenizer(test_str).input_ids

        self.assertListEqual(
            token_ids,
            [182, 117, 142, 587, 4211, 120, 117, 263, 112, 804, 109, 856, 25016, 3137, 464, 109, 26955, 3137, 1],
        )
