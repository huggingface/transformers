# coding=utf-8
# Copyright 2021 Google AI and HuggingFace Inc. team.
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

import shutil
import tempfile
import unittest

from transformers import BatchEncoding, CanineTokenizer
from transformers.file_utils import cached_property, is_torch_available

from .test_tokenization_common import TokenizerTesterMixin


if is_torch_available():
    FRAMEWORK = "pt"


class CanineTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = CanineTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()
        tokenizer = CanineTokenizer()
        tokenizer.save_pretrained(self.tmpdirname)

    @cached_property
    def canine_tokenizer(self):
        # TODO replace nielsr by google
        return CanineTokenizer.from_pretrained("nielsr/canine-s")

    def get_tokenizer(self, **kwargs) -> CanineTokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def test_prepare_batch_integration(self):
        tokenizer = self.canine_tokenizer
        src_text = ["Life is like a box of chocolates.", "You never know what you're gonna get."]
        # fmt: off
        expected_src_tokens = [57344, 76, 105, 102, 101, 32, 105, 115, 32, 108, 105, 107, 101, 32, 97, 32, 98, 111, 120, 32, 111, 102, 32, 99, 104, 111, 99, 111, 108, 97, 116, 101, 115, 46, 57345, 0, 0, 0, 0]
        # fmt: on
        batch = tokenizer(src_text, padding=True, return_tensors=FRAMEWORK)
        self.assertIsInstance(batch, BatchEncoding)

        result = list(batch.input_ids.numpy()[0])

        self.assertListEqual(expected_src_tokens, result)

        self.assertEqual((2, 39), batch.input_ids.shape)
        self.assertEqual((2, 39), batch.attention_mask.shape)

    def test_encoding_keys(self):
        tokenizer = self.canine_tokenizer
        src_text = ["Once there was a man.", "He wrote a test in HuggingFace Tranformers."]
        batch = tokenizer(src_text, padding=True, return_tensors=FRAMEWORK)
        # check if input_ids, attention_mask and token_type_ids are returned 
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("token_type_ids", batch)

    def test_max_length_integration(self):
        tokenizer = self.canine_tokenizer
        tgt_text = [
            "What's the weater?",
            "It's about 25 degrees.",
        ]
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                tgt_text, max_length=32, padding="max_length", truncation=True, return_tensors=FRAMEWORK
            )
        self.assertEqual(32, targets["input_ids"].shape[1])

    # cannot use default save_and_load_tokenzier test method because tokenzier has no vocab
    def test_save_and_load_tokenizer(self):
        # safety check on max_len default value so we are sure the test works
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                self.assertNotEqual(tokenizer.model_max_length, 42)

        # Now let's start the test
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                self.assertListEqual(before_tokens, after_tokens)

                shutil.rmtree(tmpdirname)

        tokenizers = self.get_tokenizers(model_max_length=42)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                tokenizer.add_tokens(["bim", "bambam"])
                additional_special_tokens = tokenizer.additional_special_tokens
                additional_special_tokens.append("new_additional_special_token")
                tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
                before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                self.assertListEqual(before_tokens, after_tokens)
                self.assertIn("new_additional_special_token", after_tokenizer.additional_special_tokens)
                self.assertEqual(after_tokenizer.model_max_length, 42)

                tokenizer = tokenizer.__class__.from_pretrained(tmpdirname, model_max_length=43)
                self.assertEqual(tokenizer.model_max_length, 43)

                shutil.rmtree(tmpdirname)

    # tokenizer can be instantiated without any pretrained files, so no need for pretrained tokenizer list
    def test_pretrained_model_lists(self):
        pass

    # tokenizer does not have vocabulary
    def test_get_vocab(self):
        pass

    # inputs cannot be pretokenized since ids depend on whole input string and not just on single characters
    def test_pretokenized_inputs(self):
        pass

    # tests all ids in vocab => vocab doesn't exist so unnecessary to test
    def test_conversion_reversible(self):
        pass