# coding=utf-8
# Copyright 2020 Google T5 Authors and HuggingFace Inc. team.
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

from transformers import BatchEncoding, ByT5Tokenizer
from transformers.file_utils import cached_property, is_tf_available, is_torch_available

from .test_tokenization_common import TokenizerTesterMixin


if is_torch_available():
    FRAMEWORK = "pt"
elif is_tf_available():
    FRAMEWORK = "tf"
else:
    FRAMEWORK = "jax"


class ByT5TokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = ByT5Tokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()
        tokenizer = ByT5Tokenizer()
        tokenizer.save_pretrained(self.tmpdirname)

    @cached_property
    def t5_base_tokenizer(self):
        return ByT5Tokenizer.from_pretrained("google/byt5-small")

    def get_tokenizer(self, **kwargs) -> ByT5Tokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def test_eos_treatment(self):
        tokenizer = self.t5_base_tokenizer
        batch_with_eos_added = tokenizer(["hi</s>", "I went to the gym</s>", "</s>"])
        batch_without_eos_added = tokenizer(["hi", "I went to the gym", ""])
        self.assertListEqual(batch_with_eos_added["input_ids"], batch_without_eos_added["input_ids"])

    def test_prepare_batch_integration(self):
        tokenizer = self.t5_base_tokenizer
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        # fmt: off
        expected_src_tokens = [68, 35, 111, 114, 113, 106, 35, 115, 100, 117, 100, 106, 117, 100, 115, 107, 35, 105, 114, 117, 35, 118, 120, 112, 112, 100, 117, 108, 125, 100, 119, 108, 114, 113, 49, 1, 0]
        # fmt: on
        batch = tokenizer(src_text, padding=True, return_tensors=FRAMEWORK)
        self.assertIsInstance(batch, BatchEncoding)

        if FRAMEWORK != "jax":
            result = list(batch.input_ids.numpy()[0])
        else:
            result = list(batch.input_ids.tolist()[0])

        self.assertListEqual(expected_src_tokens, result)

        self.assertEqual((2, 37), batch.input_ids.shape)
        self.assertEqual((2, 37), batch.attention_mask.shape)

    def test_empty_target_text(self):
        tokenizer = self.t5_base_tokenizer
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        batch = tokenizer(src_text, padding=True, return_tensors=FRAMEWORK)
        # check if input_ids are returned and no decoder_input_ids
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertNotIn("decoder_input_ids", batch)
        self.assertNotIn("decoder_attention_mask", batch)

    def test_max_length_integration(self):
        tokenizer = self.t5_base_tokenizer
        tgt_text = [
            "Summary of the text.",
            "Another summary.",
        ]
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                tgt_text, max_length=32, padding="max_length", truncation=True, return_tensors=FRAMEWORK
            )
        self.assertEqual(32, targets["input_ids"].shape[1])

    def test_eos_in_input(self):
        tokenizer = self.t5_base_tokenizer
        src_text = ["A long paragraph for summarization. </s>"]
        tgt_text = ["Summary of the text. </s>"]
        # fmt: off
        expected_src_tokens = [68, 35, 111, 114, 113, 106, 35, 115, 100, 117, 100, 106, 117, 100, 115, 107, 35, 105, 114, 117, 35, 118, 120, 112, 112, 100, 117, 108, 125, 100, 119, 108, 114, 113, 49, 35, 1]
        expected_tgt_tokens = [86, 120, 112, 112, 100, 117, 124, 35, 114, 105, 35, 119, 107, 104, 35, 119, 104, 123, 119, 49, 35, 1]
        # fmt: on

        batch = tokenizer(src_text)
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(tgt_text)

        self.assertEqual(expected_src_tokens, batch["input_ids"][0])
        self.assertEqual(expected_tgt_tokens, targets["input_ids"][0])

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
