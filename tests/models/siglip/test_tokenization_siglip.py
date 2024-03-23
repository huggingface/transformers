# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

from transformers import SPIECE_UNDERLINE, AddedToken, BatchEncoding, SiglipTokenizer
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
class SiglipTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google/siglip-base-patch16-224"
    tokenizer_class = SiglipTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True
    test_sentencepiece_ignore_case = True

    # Copied from tests.models.t5.test_tokenization_t5.T5TokenizationTest.setUp with T5->Siglip
    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = SiglipTokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    # Copied from tests.models.t5.test_tokenization_t5.T5TokenizationTest.test_convert_token_and_id with T5->Siglip
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

    def test_full_tokenizer(self):
        tokenizer = SiglipTokenizer(SAMPLE_VOCAB)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁this", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [66, 46, 10, 170, 382])

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE,
                "i",
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
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "é",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(ids, [7, 23, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 12, 66, 46, 72, 80, 6, 0])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                SPIECE_UNDERLINE,
                "i",
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
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "<unk>",
            ],
        )

    @cached_property
    def siglip_tokenizer(self):
        return SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224")

    # Copied from tests.models.t5.test_tokenization_t5.T5TokenizationTest.get_tokenizer with T5->Siglip
    def get_tokenizer(self, **kwargs) -> SiglipTokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    # Copied from tests.models.t5.test_tokenization_t5.T5TokenizationTest.test_rust_and_python_full_tokenizers with T5->Siglip
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
        tokenizer = self.siglip_tokenizer
        batch_with_eos_added = tokenizer(["hi</s>", "I went to the gym</s>", "</s>"])
        batch_without_eos_added = tokenizer(["hi", "I went to the gym", ""])
        self.assertListEqual(batch_with_eos_added["input_ids"], batch_without_eos_added["input_ids"])

    def test_prepare_batch(self):
        tokenizer = self.siglip_tokenizer
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        expected_src_tokens = [262, 266, 476, 8532, 270, 4460, 3949, 1682, tokenizer.eos_token_id]
        batch = tokenizer(src_text, padding=True, return_tensors=FRAMEWORK)
        self.assertIsInstance(batch, BatchEncoding)

        if FRAMEWORK != "jax":
            result = list(batch.input_ids.numpy()[0])
        else:
            result = list(batch.input_ids.tolist()[0])

        self.assertListEqual(expected_src_tokens, result)

        self.assertEqual((2, 9), batch.input_ids.shape)

    def test_empty_target_text(self):
        tokenizer = self.siglip_tokenizer
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        batch = tokenizer(src_text, padding=True, return_tensors=FRAMEWORK)
        # check if input_ids are returned and no decoder_input_ids
        self.assertIn("input_ids", batch)
        self.assertNotIn("decoder_input_ids", batch)
        self.assertNotIn("decoder_attention_mask", batch)

    def test_max_length(self):
        tokenizer = self.siglip_tokenizer
        tgt_text = ["Summary of the text.", "Another summary."]
        targets = tokenizer(
            text_target=tgt_text, max_length=32, padding="max_length", truncation=True, return_tensors=FRAMEWORK
        )
        self.assertEqual(32, targets["input_ids"].shape[1])

    def test_eos_in_input(self):
        tokenizer = self.siglip_tokenizer
        src_text = ["A long paragraph for summarization. </s>"]
        tgt_text = ["Summary of the text. </s>"]
        expected_src_tokens = [262, 266, 476, 8532, 270, 4460, 3949, 1682, 1]
        expected_tgt_tokens = [6254, 267, 260, 1443, 1]

        batch = tokenizer(src_text, text_target=tgt_text)

        self.assertEqual(expected_src_tokens, batch["input_ids"][0])
        self.assertEqual(expected_tgt_tokens, batch["labels"][0])

    @unittest.skip(reason="SiglipTokenizer strips the punctuation")
    def test_subword_regularization_tokenizer(self):
        pass

    @unittest.skip(reason="SiglipTokenizer strips the punctuation")
    def test_pickle_subword_regularization_tokenizer(self):
        pass

    # Copied from tests.models.t5.test_tokenization_t5.T5TokenizationTest.test_special_tokens_initialization with T5->Siglip
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

    # Copied from tests.models.t5.test_tokenization_t5.T5TokenizationTest.test_special_tokens_initialization_with_non_empty_additional_special_tokens with T5->Siglip
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
                # self.assertIn("an_additional_special_token",tokenizer_without_change_in_init.get_vocab()) # BySiglipTokenization no vocab
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

    def test_sentencepiece_tokenize_and_convert_tokens_to_string(self):
        """Test ``_tokenize`` and ``convert_tokens_to_string``."""
        if not self.test_sentencepiece:
            return

        tokenizer = self.get_tokenizer()
        text = "This is text to test the tokenizer."

        if self.test_sentencepiece_ignore_case:
            text = text.lower()

        tokens = tokenizer.tokenize(text)

        self.assertTrue(len(tokens) > 0)

        # check if converting back to original text works
        reverse_text = tokenizer.convert_tokens_to_string(tokens)

        if self.test_sentencepiece_ignore_case:
            reverse_text = reverse_text.lower()

        expected_text = "this is text to test the tokenizer"
        self.assertEqual(reverse_text, expected_text)

        special_tokens = tokenizer.all_special_tokens
        special_tokens_string = tokenizer.convert_tokens_to_string(special_tokens)
        for special_token in special_tokens:
            self.assertIn(special_token, special_tokens_string)

        if self.test_rust_tokenizer:
            rust_tokenizer = self.get_rust_tokenizer()
            special_tokens_string_rust = rust_tokenizer.convert_tokens_to_string(special_tokens)
            self.assertEqual(special_tokens_string, special_tokens_string_rust)

    # overwritten from `test_tokenization_common` since Siglip has no max length
    # Copied from tests.models.t5.test_tokenization_t5.T5TokenizationTest.test_pretrained_model_lists with T5->Siglip
    def test_pretrained_model_lists(self):
        # We should have at least one default checkpoint for each tokenizer
        # We should specify the max input length as well (used in some part to list the pretrained checkpoints)
        self.assertGreaterEqual(len(self.tokenizer_class.pretrained_vocab_files_map), 1)
        self.assertGreaterEqual(len(list(self.tokenizer_class.pretrained_vocab_files_map.values())[0]), 1)

    @slow
    def test_tokenizer_integration(self):
        tokenizer = SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224")

        # fmt: off
        texts = [
            'the real mountain view',
            'Zürich',
            'San Francisco',
            'a picture of a laptop with the lockscreen on, a cup of cappucino, salt and pepper grinders. The view through the window reveals lake Zürich and the Alps in the background of the city.',
        ]

        expected_input_ids = [
            [260, 638, 3293, 870, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [262, 761, 5879, 5345, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [262, 264, 452, 20563, 15949, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [262, 266, 1357, 267, 262, 266, 4429, 275, 260, 3940, 6360, 277, 262, 266, 3064, 267, 3549, 388, 16538, 296, 298, 2617, 263, 4869, 14998, 264, 260, 870, 393, 260, 1710, 7958, 4324, 262, 761, 5879, 5345, 263, 260, 1518, 388, 264, 268, 260, 1970, 267, 260, 741, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        # fmt: on

        for text, expected in zip(texts, expected_input_ids):
            input_ids = tokenizer(text, padding="max_length").input_ids
            self.assertListEqual(input_ids, expected)

    def test_some_edge_cases(self):
        tokenizer = SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224", legacy=False)

        sp_tokens = tokenizer.sp_model.encode("</s>>", out_type=str)
        self.assertEqual(sp_tokens, ["</", "s", ">", ">"])
        tokens = tokenizer.tokenize("</s>>")
        self.assertNotEqual(sp_tokens, tokens)
        self.assertEqual(tokens, ["</s>"])

        tokens = tokenizer.tokenize("")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, tokenizer.sp_model.encode("", out_type=str))

        tokens = tokenizer.tokenize(" ")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, tokenizer.sp_model.encode(" ", out_type=str))

        tokens = tokenizer.tokenize("▁")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, tokenizer.sp_model.encode("▁", out_type=str))

        tokens = tokenizer.tokenize(" ▁")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, tokenizer.sp_model.encode("▁", out_type=str))


@require_sentencepiece
@require_tokenizers
class CommonSpmIntegrationTests(unittest.TestCase):
    """
    A class that regroups important test to make sure that we properly handle the special tokens.
    """

    @classmethod
    def setUpClass(cls):
        tokenizer = SiglipTokenizer(SAMPLE_VOCAB, extra_ids=0, legacy=False)
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [AddedToken("<extra_id_0>", rstrip=False, lstrip=False)]}
        )
        cls.tokenizer = tokenizer

    def test_add_dummy_prefix(self):
        # make sure `'▁'` is prepended, and outputs match sp_model's
        # `sentencepiece.NormalizerSpec.add_dummy_prefix` attribute
        input_ids = self.tokenizer.encode(". Hello", add_special_tokens=False)
        self.assertEqual(input_ids, [37, 86, 20])
        self.assertEqual(input_ids, [37, 86, 20])
        tokens = self.tokenizer.tokenize(". Hello")
        self.assertEqual(tokens, ["▁he", "ll", "o"])

        tokens = self.tokenizer.tokenize("")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, self.tokenizer.sp_model.encode("", out_type=str))

        tokens = self.tokenizer.tokenize(" ")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, self.tokenizer.sp_model.encode(" ", out_type=str))

        tokens = self.tokenizer.tokenize("▁")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, self.tokenizer.sp_model.encode("▁", out_type=str))

    def test_remove_extra_whitespaces(self):
        # make sure the extra spaces are eaten
        # sentencepiece.NormalizerSpec.remove_extra_whitespaces attribute
        input_ids = self.tokenizer.encode("       . Hello", add_special_tokens=False)
        self.assertEqual(input_ids, [37, 86, 20])
        self.assertEqual(input_ids, [37, 86, 20])
        tokens = self.tokenizer.tokenize(" . Hello")
        self.assertEqual(tokens, ["▁he", "ll", "o"])

        # `'▁'` is also a whitespace
        input_ids = self.tokenizer.encode("▁He is not")
        self.assertEqual(input_ids, [37, 46, 44, 2])
        tokens = self.tokenizer.tokenize("▁He is not")
        self.assertEqual(tokens, ["▁he", "▁is", "▁not"])  # no extra space added

        input_ids = self.tokenizer.encode("▁He is not             ▁He")
        self.assertEqual(input_ids, [37, 46, 44, 37, 2])
        tokens = self.tokenizer.tokenize("▁He is not              ▁He")
        self.assertEqual(tokens, ["▁he", "▁is", "▁not", "▁he"])  # spaces are eaten by spm even if not start
