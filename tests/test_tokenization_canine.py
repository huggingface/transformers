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

import json
import os
import shutil
import tempfile
import unittest

from transformers import BatchEncoding, CanineTokenizer
from transformers.file_utils import cached_property
from transformers.testing_utils import require_tokenizers, require_torch
from transformers.tokenization_utils import AddedToken

from .test_tokenization_common import TokenizerTesterMixin


class CanineTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = CanineTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()
        tokenizer = CanineTokenizer()
        tokenizer.save_pretrained(self.tmpdirname)

    @cached_property
    def canine_tokenizer(self):
        return CanineTokenizer.from_pretrained("google/canine-s")

    def get_tokenizer(self, **kwargs) -> CanineTokenizer:
        tokenizer = self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)
        tokenizer._unicode_vocab_size = 1024
        return tokenizer

    @require_torch
    def test_prepare_batch_integration(self):
        tokenizer = self.canine_tokenizer
        src_text = ["Life is like a box of chocolates.", "You never know what you're gonna get."]
        # fmt: off
        expected_src_tokens = [57344, 76, 105, 102, 101, 32, 105, 115, 32, 108, 105, 107, 101, 32, 97, 32, 98, 111, 120, 32, 111, 102, 32, 99, 104, 111, 99, 111, 108, 97, 116, 101, 115, 46, 57345, 0, 0, 0, 0]
        # fmt: on
        batch = tokenizer(src_text, padding=True, return_tensors="pt")
        self.assertIsInstance(batch, BatchEncoding)

        result = list(batch.input_ids.numpy()[0])

        self.assertListEqual(expected_src_tokens, result)

        self.assertEqual((2, 39), batch.input_ids.shape)
        self.assertEqual((2, 39), batch.attention_mask.shape)

    @require_torch
    def test_encoding_keys(self):
        tokenizer = self.canine_tokenizer
        src_text = ["Once there was a man.", "He wrote a test in HuggingFace Tranformers."]
        batch = tokenizer(src_text, padding=True, return_tensors="pt")
        # check if input_ids, attention_mask and token_type_ids are returned
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("token_type_ids", batch)

    @require_torch
    def test_max_length_integration(self):
        tokenizer = self.canine_tokenizer
        tgt_text = [
            "What's the weater?",
            "It's about 25 degrees.",
        ]
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(tgt_text, max_length=32, padding="max_length", truncation=True, return_tensors="pt")
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

                additional_special_tokens = tokenizer.additional_special_tokens

                # We can add a new special token for Canine as follows:
                new_additional_special_token = chr(0xE007)
                additional_special_tokens.append(new_additional_special_token)
                tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
                before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                self.assertListEqual(before_tokens, after_tokens)
                self.assertIn(new_additional_special_token, after_tokenizer.additional_special_tokens)
                self.assertEqual(after_tokenizer.model_max_length, 42)

                tokenizer = tokenizer.__class__.from_pretrained(tmpdirname, model_max_length=43)
                self.assertEqual(tokenizer.model_max_length, 43)

                shutil.rmtree(tmpdirname)

    def test_add_special_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, ids = self.get_clean_sequence(tokenizer)

                # a special token for Canine can be defined as follows:
                SPECIAL_TOKEN = 0xE005
                special_token = chr(SPECIAL_TOKEN)

                tokenizer.add_special_tokens({"cls_token": special_token})
                encoded_special_token = tokenizer.encode(special_token, add_special_tokens=False)
                self.assertEqual(len(encoded_special_token), 1)

                text = tokenizer.decode(ids + encoded_special_token, clean_up_tokenization_spaces=False)
                encoded = tokenizer.encode(text, add_special_tokens=False)

                input_encoded = tokenizer.encode(input_text, add_special_tokens=False)
                special_token_id = tokenizer.encode(special_token, add_special_tokens=False)
                self.assertEqual(encoded, input_encoded + special_token_id)

                decoded = tokenizer.decode(encoded, skip_special_tokens=True)
                self.assertTrue(special_token not in decoded)

    def test_tokenize_special_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                SPECIAL_TOKEN_1 = chr(0xE005)
                SPECIAL_TOKEN_2 = chr(0xE006)

                # `add_tokens` method stores special tokens only in `tokenizer.unique_no_split_tokens`. (in tokenization_utils.py)
                tokenizer.add_tokens([SPECIAL_TOKEN_1], special_tokens=True)
                # `add_special_tokens` method stores special tokens in `tokenizer.additional_special_tokens`,
                # which also occur in `tokenizer.all_special_tokens`. (in tokenization_utils_base.py)
                tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKEN_2]})

                token_1 = tokenizer.tokenize(SPECIAL_TOKEN_1)
                token_2 = tokenizer.tokenize(SPECIAL_TOKEN_2)

                self.assertEqual(len(token_1), 1)
                self.assertEqual(len(token_2), 1)
                self.assertEqual(token_1[0], SPECIAL_TOKEN_1)
                self.assertEqual(token_2[0], SPECIAL_TOKEN_2)

    @require_tokenizers
    def test_added_token_serializable(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                # a special token for Canine can be defined as follows:
                NEW_TOKEN = 0xE006
                new_token = chr(NEW_TOKEN)

                new_token = AddedToken(new_token, lstrip=True)
                tokenizer.add_special_tokens({"additional_special_tokens": [new_token]})

                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    tokenizer.save_pretrained(tmp_dir_name)
                    tokenizer.from_pretrained(tmp_dir_name)

    def test_special_tokens_initialization_with_non_empty_additional_special_tokens(self):
        tokenizer_list = []
        if self.test_slow_tokenizer:
            tokenizer_list.append((self.tokenizer_class, self.get_tokenizer()))

        if self.test_rust_tokenizer:
            tokenizer_list.append((self.rust_tokenizer_class, self.get_rust_tokenizer()))

        for tokenizer_class, tokenizer_utils in tokenizer_list:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer_utils.save_pretrained(tmp_dir)

                with open(os.path.join(tmp_dir, "special_tokens_map.json"), encoding="utf-8") as json_file:
                    special_tokens_map = json.load(json_file)

                with open(os.path.join(tmp_dir, "tokenizer_config.json"), encoding="utf-8") as json_file:
                    tokenizer_config = json.load(json_file)

                # a special token for Canine can be defined as follows:
                NEW_TOKEN = 0xE006
                new_token_1 = chr(NEW_TOKEN)

                special_tokens_map["additional_special_tokens"] = [new_token_1]
                tokenizer_config["additional_special_tokens"] = [new_token_1]

                with open(os.path.join(tmp_dir, "special_tokens_map.json"), "w", encoding="utf-8") as outfile:
                    json.dump(special_tokens_map, outfile)
                with open(os.path.join(tmp_dir, "tokenizer_config.json"), "w", encoding="utf-8") as outfile:
                    json.dump(tokenizer_config, outfile)

                # the following checks allow us to verify that our test works as expected, i.e. that the tokenizer takes
                # into account the new value of additional_special_tokens given in the "tokenizer_config.json" and
                # "special_tokens_map.json" files
                tokenizer_without_change_in_init = tokenizer_class.from_pretrained(tmp_dir, extra_ids=0)
                self.assertIn(new_token_1, tokenizer_without_change_in_init.additional_special_tokens)
                # self.assertIn("an_additional_special_token",tokenizer_without_change_in_init.get_vocab()) # ByT5Tokenization no vocab
                self.assertEqual(
                    [new_token_1],
                    tokenizer_without_change_in_init.convert_ids_to_tokens(
                        tokenizer_without_change_in_init.convert_tokens_to_ids([new_token_1])
                    ),
                )

                NEW_TOKEN = 0xE007
                new_token_2 = chr(NEW_TOKEN)
                # Now we test that we can change the value of additional_special_tokens in the from_pretrained
                new_added_tokens = [AddedToken(new_token_2, lstrip=True)]
                tokenizer = tokenizer_class.from_pretrained(
                    tmp_dir, additional_special_tokens=new_added_tokens, extra_ids=0
                )

                self.assertIn(new_token_2, tokenizer.additional_special_tokens)
                # self.assertIn(new_token_2,tokenizer.get_vocab()) # ByT5Tokenization no vocab
                self.assertEqual(
                    [new_token_2], tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids([new_token_2]))
                )

    @require_tokenizers
    def test_encode_decode_with_spaces(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                input = "hello world"
                if self.space_between_special_tokens:
                    output = "[CLS] hello world [SEP]"
                else:
                    output = input
                encoded = tokenizer.encode(input, add_special_tokens=False)
                decoded = tokenizer.decode(encoded, spaces_between_special_tokens=self.space_between_special_tokens)
                self.assertIn(decoded, [output, output.lower()])

    # tokenizer has a fixed vocab_size (namely all possible unicode code points)
    def test_add_tokens_tokenizer(self):
        pass

    # CanineTokenizer does not support do_lower_case = True, as each character has its own Unicode code point
    # ("b" and "B" for example have different Unicode code points)
    def test_added_tokens_do_lower_case(self):
        pass

    # CanineModel does not support the get_input_embeddings nor the get_vocab method
    def test_np_encode_plus_sent_to_model(self):
        pass

    # CanineModel does not support the get_input_embeddings nor the get_vocab method
    def test_torch_encode_plus_sent_to_model(self):
        pass

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
