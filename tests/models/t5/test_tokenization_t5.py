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
import json
import os
import tempfile
import unittest

from transformers import SPIECE_UNDERLINE, AddedToken, BatchEncoding, T5Tokenizer, T5TokenizerFast
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, slow
from transformers.utils import cached_property, is_tf_available, is_torch_available

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")

if is_torch_available():
    FRAMEWORK = "pt"
elif is_tf_available():
    FRAMEWORK = "tf"
else:
    FRAMEWORK = "jax"


@require_sentencepiece
@require_tokenizers
class T5TokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = T5Tokenizer
    rust_tokenizer_class = T5TokenizerFast
    test_rust_tokenizer = True
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = T5Tokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<s>"
        token_id = 1

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<unk>")
        self.assertEqual(vocab_keys[1], "<s>")
        self.assertEqual(vocab_keys[-1], "<pad>")
        self.assertEqual(len(vocab_keys), 1_101)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 1_100)

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

    def test_prepare_batch(self):
        tokenizer = self.t5_base_tokenizer
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        expected_src_tokens = [71, 307, 8986, 21, 4505, 1635, 1707, 5, tokenizer.eos_token_id]
        batch = tokenizer(src_text, padding=True, return_tensors=FRAMEWORK)
        self.assertIsInstance(batch, BatchEncoding)

        if FRAMEWORK != "jax":
            result = list(batch.input_ids.numpy()[0])
        else:
            result = list(batch.input_ids.tolist()[0])

        self.assertListEqual(expected_src_tokens, result)

        self.assertEqual((2, 9), batch.input_ids.shape)
        self.assertEqual((2, 9), batch.attention_mask.shape)

    def test_empty_target_text(self):
        tokenizer = self.t5_base_tokenizer
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        batch = tokenizer(src_text, padding=True, return_tensors=FRAMEWORK)
        # check if input_ids are returned and no decoder_input_ids
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertNotIn("decoder_input_ids", batch)
        self.assertNotIn("decoder_attention_mask", batch)

    def test_max_length(self):
        tokenizer = self.t5_base_tokenizer
        tgt_text = [
            "Summary of the text.",
            "Another summary.",
        ]
        targets = tokenizer(
            text_target=tgt_text, max_length=32, padding="max_length", truncation=True, return_tensors=FRAMEWORK
        )
        self.assertEqual(32, targets["input_ids"].shape[1])

    def test_outputs_not_longer_than_maxlen(self):
        tokenizer = self.t5_base_tokenizer

        batch = tokenizer(
            ["I am a small frog" * 1000, "I am a small frog"], padding=True, truncation=True, return_tensors=FRAMEWORK
        )
        self.assertIsInstance(batch, BatchEncoding)
        # Since T5 does NOT have a max input length,
        # this test should be changed to the following in Transformers v5:
        # self.assertEqual(batch.input_ids.shape, (2, 8001))
        self.assertEqual(batch.input_ids.shape, (2, 512))

    def test_eos_in_input(self):
        tokenizer = self.t5_base_tokenizer
        src_text = ["A long paragraph for summarization. </s>"]
        tgt_text = ["Summary of the text. </s>"]
        expected_src_tokens = [71, 307, 8986, 21, 4505, 1635, 1707, 5, 1]
        expected_tgt_tokens = [20698, 13, 8, 1499, 5, 1]

        batch = tokenizer(src_text, text_target=tgt_text)

        self.assertEqual(expected_src_tokens, batch["input_ids"][0])
        self.assertEqual(expected_tgt_tokens, batch["labels"][0])

    def test_token_type_ids(self):
        src_text_1 = ["A first paragraph for summarization."]
        src_text_2 = ["A second paragraph for summarization."]

        fast_token_type_ids = self.t5_base_tokenizer_fast(
            src_text_1, src_text_2, add_special_tokens=True, return_token_type_ids=True
        ).token_type_ids
        slow_token_type_ids = self.t5_base_tokenizer(
            src_text_1, src_text_2, add_special_tokens=True, return_token_type_ids=True
        ).token_type_ids

        self.assertEqual(slow_token_type_ids, fast_token_type_ids)
        self.assertEqual(len(slow_token_type_ids[0]), 18)

    def test_fast_and_slow_same_result(self):
        src_text = "<pad> Today is <unk> nice day </s>"
        tgt_ids = [0, 1960, 19, 2, 1245, 239, 1]
        tgt_text = "<pad> Today is<unk> nice day</s>"

        fast_ids = self.t5_base_tokenizer_fast(src_text, add_special_tokens=False).input_ids
        slow_ids = self.t5_base_tokenizer(src_text, add_special_tokens=False).input_ids
        self.assertEqual(tgt_ids, fast_ids)
        self.assertEqual(tgt_ids, slow_ids)

        fast_text = self.t5_base_tokenizer_fast.decode(fast_ids)
        slow_text = self.t5_base_tokenizer.decode(fast_ids)
        self.assertEqual(tgt_text, fast_text)
        self.assertEqual(tgt_text, slow_text)

    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):

                added_tokens = [f"<extra_id_{i}>" for i in range(100)] + [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs
                )
                tokenizer_cr = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs, from_slow=True
                )
                tokenizer_p = self.tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs
                )

                p_output = tokenizer_p.encode("Hey this is a <special> token")
                r_output = tokenizer_r.encode("Hey this is a <special> token")
                cr_output = tokenizer_cr.encode("Hey this is a <special> token")

                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]

                self.assertEqual(p_output, r_output)
                self.assertEqual(cr_output, r_output)
                self.assertTrue(special_token_id in p_output)
                self.assertTrue(special_token_id in r_output)
                self.assertTrue(special_token_id in cr_output)

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

                added_tokens_extra_ids = [f"<extra_id_{i}>" for i in range(100)]

                special_tokens_map["additional_special_tokens"] = added_tokens_extra_ids + [
                    "an_additional_special_token"
                ]
                tokenizer_config["additional_special_tokens"] = added_tokens_extra_ids + [
                    "an_additional_special_token"
                ]

                with open(os.path.join(tmp_dir, "special_tokens_map.json"), "w", encoding="utf-8") as outfile:
                    json.dump(special_tokens_map, outfile)
                with open(os.path.join(tmp_dir, "tokenizer_config.json"), "w", encoding="utf-8") as outfile:
                    json.dump(tokenizer_config, outfile)

                # the following checks allow us to verify that our test works as expected, i.e. that the tokenizer takes
                # into account the new value of additional_special_tokens given in the "tokenizer_config.json" and
                # "special_tokens_map.json" files
                tokenizer_without_change_in_init = tokenizer_class.from_pretrained(
                    tmp_dir,
                )
                self.assertIn(
                    "an_additional_special_token", tokenizer_without_change_in_init.additional_special_tokens
                )
                # self.assertIn("an_additional_special_token",tokenizer_without_change_in_init.get_vocab()) # ByT5Tokenization no vocab
                self.assertEqual(
                    ["an_additional_special_token"],
                    tokenizer_without_change_in_init.convert_ids_to_tokens(
                        tokenizer_without_change_in_init.convert_tokens_to_ids(["an_additional_special_token"])
                    ),
                )

                # Now we test that we can change the value of additional_special_tokens in the from_pretrained
                new_added_tokens = added_tokens_extra_ids + [AddedToken("a_new_additional_special_token", lstrip=True)]
                tokenizer = tokenizer_class.from_pretrained(
                    tmp_dir,
                    additional_special_tokens=new_added_tokens,
                )

                self.assertIn("a_new_additional_special_token", tokenizer.additional_special_tokens)
                self.assertEqual(
                    ["a_new_additional_special_token"],
                    tokenizer.convert_ids_to_tokens(
                        tokenizer.convert_tokens_to_ids(["a_new_additional_special_token"])
                    ),
                )

    # overwritten from `test_tokenization_common` since T5 has no max length
    def test_pretrained_model_lists(self):
        # We should have at least one default checkpoint for each tokenizer
        # We should specify the max input length as well (used in some part to list the pretrained checkpoints)
        self.assertGreaterEqual(len(self.tokenizer_class.pretrained_vocab_files_map), 1)
        self.assertGreaterEqual(len(list(self.tokenizer_class.pretrained_vocab_files_map.values())[0]), 1)

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {'input_ids': [[31220, 7, 41, 14034, 801, 38, 3, 102, 63, 17, 127, 524, 18, 7031, 2032, 277, 11, 3, 102, 63, 17, 127, 524, 18, 2026, 17, 10761, 18, 7041, 61, 795, 879, 18, 19681, 4648, 7, 41, 12920, 382, 6, 350, 6383, 4949, 6, 2158, 12920, 382, 9, 6, 3, 4, 11160, 6, 2043, 17153, 279, 49, 17, 6, 3, 4, 434, 9688, 11439, 21, 6869, 10509, 17725, 41, 567, 9138, 61, 11, 6869, 10509, 11946, 41, 18207, 517, 61, 28, 147, 3538, 1220, 7140, 10761, 2250, 16, 910, 1220, 8024, 11, 1659, 1413, 32, 883, 2020, 344, 2215, 226, 6, 12901, 382, 127, 524, 11, 4738, 7, 127, 15390, 5, 1], [272, 24203, 19, 876, 12, 554, 18, 9719, 1659, 2647, 26352, 6497, 7, 45, 73, 9339, 400, 26, 1499, 57, 22801, 10760, 30, 321, 646, 11, 269, 2625, 16, 66, 7500, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [37, 1704, 4216, 3, 20400, 4418, 7, 147, 8, 19743, 1782, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # noqa: E501
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="t5-base",
            revision="5a7ff2d8f5117c194c7e32ec1ccbf04642cca99b",
        )
