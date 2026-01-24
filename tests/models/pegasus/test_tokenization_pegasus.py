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
from functools import cached_property

from transformers import PegasusTokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, require_torch, slow
from transformers.tokenization_utils_sentencepiece import SentencePieceExtractor

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_no_bos.model")


@require_sentencepiece
@require_tokenizers
class PegasusTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    # TokenizerTesterMixin configuration
    from_pretrained_id = ["google/pegasus-xsum"]
    tokenizer_class = PegasusTokenizer

    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 's', 'é', '.', '▁', '生活的真谛是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '▁', '<', 's', '>', '▁hi', '<', 's', '>', 'there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁Hello', '.', '▁But', '▁i', 'rd', '▁and', '▁', 'ปี', '▁i', 'rd', '▁', 'ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [182, 117, 114, 804, 110, 105, 125, 140, 1723, 115, 950, 15337, 108, 111, 136, 117, 54154, 116, 5371, 107, 110, 105, 4451, 8087, 4451, 8087, 8087, 110, 105, 116, 2314, 9800, 105, 116, 2314, 7731, 139, 645, 4211, 246, 129, 2023, 33041, 151, 8087, 107, 343, 532, 2007, 111, 110, 105, 532, 2007, 110, 105, 10532, 199, 127, 119, 557]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁a', '▁test', '▁', '<unk>', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 's', 'é', '.', '▁', '<unk>', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '▁', '<unk>', 's', '>', '▁hi', '<unk>', 's', '>', 'there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁Hello', '.', '▁But', '▁i', 'rd', '▁and', '▁', '<unk>', '▁i', 'rd', '▁', '<unk>', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test <unk> I was born in 92000, and this is falsé. <unk> Hi Hello Hi Hello Hello <unk>s> hi<unk>s>there The following string should be properly encoded: Hello. But ird and <unk> ird <unk> Hey how are you doing"

    @cached_property
    def _large_tokenizer(self):
        return PegasusTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

    @unittest.skip(reason="Test expects BigBird-Pegasus-specific vocabulary and special tokens")
    def test_large_mask_tokens(self):
        tokenizer = self._large_tokenizer
        # <mask_1> masks whole sentence while <mask_2> masks single word
        raw_input_str = "<mask_1> To ensure a <mask_2> flow of bank resolutions."
        desired_result = [2, 413, 615, 114, 3, 1971, 113, 1679, 10710, 107, 1]
        ids = tokenizer([raw_input_str], return_tensors=None).input_ids[0]
        self.assertListEqual(desired_result, ids)

    @unittest.skip(reason="Test expects BigBird-Pegasus-specific vocabulary")
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

    @unittest.skip(reason="Test expects BigBird-Pegasus-specific vocabulary")
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
        expected_encoding = {'input_ids': [[38979, 143, 18485, 606, 130, 26669, 87686, 121, 54189, 1129, 111, 26669, 87686, 121, 9114, 14787, 121, 13249, 158, 592, 956, 121, 14621, 31576, 143, 62613, 108, 9688, 930, 43430, 11562, 62613, 304, 108, 11443, 897, 108, 9314, 17415, 63399, 108, 11443, 7614, 18316, 118, 4284, 7148, 12430, 143, 1400, 25703, 158, 111, 4284, 7148, 11772, 143, 21297, 1064, 158, 122, 204, 3506, 1754, 1133, 14787, 1581, 115, 33224, 4482, 111, 1355, 110, 29173, 317, 50833, 108, 20147, 94665, 111, 77198, 107, 1], [110, 62613, 117, 638, 112, 1133, 121, 20098, 1355, 79050, 13872, 135, 1596, 53541, 1352, 141, 13039, 5542, 124, 302, 518, 111, 268, 2956, 115, 149, 4427, 107, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [139, 1235, 2799, 18289, 17780, 204, 109, 9474, 1296, 107, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # fmt: skip

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="google/bigbird-pegasus-large-arxiv",
            revision="ba85d0851d708441f91440d509690f1ab6353415",
        )


@require_sentencepiece
@require_tokenizers
class BigBirdPegasusTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google/pegasus-xsum"
    tokenizer_class = PegasusTokenizer
    test_rust_tokenizer = True
    test_sentencepiece = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # We have a SentencePiece fixture for testing
        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        _, vocab_scores, _ = extractor.extract()
        tokenizer = PegasusTokenizer(vocab=vocab_scores, offset=0, mask_token_sent=None, mask_token="[MASK]")
        tokenizer.save_pretrained(cls.tmpdirname)

    @cached_property
    def _large_tokenizer(self):
        return PegasusTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

    @classmethod
    def get_tokenizer(cls, pretrained_name=None, **kwargs) -> PegasusTokenizer:
        pretrained_name = pretrained_name or cls.tmpdirname
        return PegasusTokenizer.from_pretrained(pretrained_name, **kwargs)

    def get_input_output_texts(self, tokenizer):
        return ("This is a test", "This is a test")

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
        test_str = (
            "This is an example string that is used to test the original TF implementation against the HF"
            " implementation"
        )

        token_ids = self._large_tokenizer(test_str).input_ids

        self.assertListEqual(
            token_ids,
            [182, 117, 142, 587, 4211, 120, 117, 263, 112, 804, 109, 856, 25016, 3137, 464, 109, 26955, 3137, 1],
        )
