# coding=utf-8
# Copyright 2020 Ecole Polytechnique and HuggingFace Inc. team.
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

import pickle
import unittest

from transformers import BarthezTokenizer, BarthezTokenizerFast, BatchEncoding
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow

from .test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
@require_sentencepiece
@slow  # see https://github.com/huggingface/transformers/issues/11457
class BarthezTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BarthezTokenizer
    rust_tokenizer_class = BarthezTokenizerFast
    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        tokenizer = BarthezTokenizerFast.from_pretrained("moussaKam/mbarthez")
        tokenizer.save_pretrained(self.tmpdirname)
        tokenizer.save_pretrained(self.tmpdirname, legacy_format=False)
        self.tokenizer = tokenizer

    @require_torch
    def test_prepare_batch(self):
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        expected_src_tokens = [0, 57, 3018, 70307, 91, 2]

        batch = self.tokenizer(
            src_text, max_length=len(expected_src_tokens), padding=True, truncation=True, return_tensors="pt"
        )
        self.assertIsInstance(batch, BatchEncoding)

        self.assertEqual((2, 6), batch.input_ids.shape)
        self.assertEqual((2, 6), batch.attention_mask.shape)
        result = batch.input_ids.tolist()[0]
        self.assertListEqual(expected_src_tokens, result)

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

    def test_subword_regularization_tokenizer(self):
        # Subword regularization is only available for the slow tokenizer.
        tokenizer = self.tokenizer_class.from_pretrained(
            "moussaKam/mbarthez", sp_model_kwargs={"enable_sampling": True, "alpha": 0.1, "nbest_size": -1}
        )

        self.check_subword_sampling(tokenizer)

    def test_pickle_subword_regularization_tokenizer(self):
        """Google pickle __getstate__ __setstate__ if you are struggling with this."""
        # Subword regularization is only available for the slow tokenizer.
        sp_model_kwargs = {"enable_sampling": True, "alpha": 0.1, "nbest_size": -1}
        tokenizer = self.tokenizer_class.from_pretrained("moussaKam/mbarthez", sp_model_kwargs=sp_model_kwargs)
        tokenizer_bin = pickle.dumps(tokenizer)
        del tokenizer
        tokenizer_new = pickle.loads(tokenizer_bin)

        self.assertIsNotNone(tokenizer_new.sp_model_kwargs)
        self.assertTrue(isinstance(tokenizer_new.sp_model_kwargs, dict))
        self.assertEqual(tokenizer_new.sp_model_kwargs, sp_model_kwargs)
        self.check_subword_sampling(tokenizer_new)
