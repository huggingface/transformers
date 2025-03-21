# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import inspect
import pickle
import shutil
import tempfile
import unittest

from transformers import (
    SPIECE_UNDERLINE,
    AddedToken,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    SpecialTokensMixin,
)
from transformers.convert_slow_tokenizer import MoshiConverter
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
)

from ...test_tokenization_common import SMALL_TRAINING_CORPUS, TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class MoshiTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = ["kmhf/hf-moshiko"]
    rust_tokenizer_class = PreTrainedTokenizerFast

    test_slow_tokenizer = False
    test_rust_tokenizer = True
    from_pretrained_kwargs = {}

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=MoshiConverter(vocab_file=SAMPLE_VOCAB).converted(),
            bos_token="<s>",
            unk_token="<unk>",
            eos_token="</s>",
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(self.tmpdirname)

    def get_rust_tokenizer(self, **kwargs) -> PreTrainedTokenizerFast:
        return self.rust_tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    @unittest.skip(reason="No slow tokenizer")
    def test_added_tokens_serialization(self):
        pass

    @unittest.skip(reason="PreTrainedTokenizerFast doesn't have tokenizer_file in its signature")
    def test_rust_tokenizer_signature(self):
        pass

    @unittest.skip(reason="No slow tokenizer")
    def test_encode_decode_with_spaces(self):
        pass

    def test_full_tokenizer(self):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=MoshiConverter(vocab_file=SAMPLE_VOCAB).converted(),
            bos_token="<s>",
            unk_token="<unk>",
            eos_token="</s>",
        )

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [285, 46, 10, 170, 382],
        )

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
        self.assertListEqual(
            ids,
            [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 0, 4],
        )

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

    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                added_tokens = [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs
                )
                r_output = tokenizer_r.encode("Hey this is a <special> token")

                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]

                self.assertTrue(special_token_id in r_output)

    def test_picklable(self):
        with tempfile.NamedTemporaryFile() as f:
            shutil.copyfile(SAMPLE_VOCAB, f.name)
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=MoshiConverter(vocab_file=f.name).converted(),
                bos_token="<s>",
                unk_token="<unk>",
                eos_token="</s>",
            )
            pickled_tokenizer = pickle.dumps(tokenizer)
        pickle.loads(pickled_tokenizer)

    def test_training_new_tokenizer(self):
        # This feature only exists for fast tokenizers
        if not self.test_rust_tokenizer:
            self.skipTest(reason="test_rust_tokenizer is set to False")

        tokenizer = self.get_rust_tokenizer()
        new_tokenizer = tokenizer.train_new_from_iterator(SMALL_TRAINING_CORPUS, 100)

        # Test we can use the new tokenizer with something not seen during training
        inputs = new_tokenizer(["This is the first sentence", "This sentence is different 🤗."])
        self.assertEqual(len(inputs["input_ids"]), 2)
        decoded_input = new_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        expected_result = "This is the first sentence"

        self.assertEqual(expected_result, decoded_input)

        # We check that the parameters of the tokenizer remained the same
        # Check we have the same number of added_tokens for both pair and non-pair inputs.
        self.assertEqual(tokenizer.num_special_tokens_to_add(False), new_tokenizer.num_special_tokens_to_add(False))
        self.assertEqual(tokenizer.num_special_tokens_to_add(True), new_tokenizer.num_special_tokens_to_add(True))

        # Check we have the correct max_length for both pair and non-pair inputs.
        self.assertEqual(tokenizer.max_len_single_sentence, new_tokenizer.max_len_single_sentence)
        self.assertEqual(tokenizer.max_len_sentences_pair, new_tokenizer.max_len_sentences_pair)

        # Assert the set of special tokens match as we didn't ask to change them
        self.assertSequenceEqual(
            tokenizer.all_special_tokens_extended,
            new_tokenizer.all_special_tokens_extended,
        )

        self.assertDictEqual(tokenizer.special_tokens_map, new_tokenizer.special_tokens_map)

    def test_training_new_tokenizer_with_special_tokens_change(self):
        # This feature only exists for fast tokenizers
        if not self.test_rust_tokenizer:
            self.skipTest(reason="test_rust_tokenizer is set to False")

        tokenizer = self.get_rust_tokenizer()
        # Test with a special tokens map
        class_signature = inspect.signature(tokenizer.__class__)
        if "cls_token" in class_signature.parameters:
            new_tokenizer = tokenizer.train_new_from_iterator(
                SMALL_TRAINING_CORPUS, 100, special_tokens_map={tokenizer.cls_token: "<cls>"}
            )
            cls_id = new_tokenizer.get_vocab()["<cls>"]
            self.assertEqual(new_tokenizer.cls_token, "<cls>")
            self.assertEqual(new_tokenizer.cls_token_id, cls_id)

        # Create a new mapping from the special tokens defined in the original tokenizer
        special_tokens_list = SpecialTokensMixin.SPECIAL_TOKENS_ATTRIBUTES.copy()
        special_tokens_list.remove("additional_special_tokens")
        special_tokens_map = {}
        for token in special_tokens_list:
            # Get the private one to avoid unnecessary warnings.
            if getattr(tokenizer, token) is not None:
                special_token = getattr(tokenizer, token)
                special_tokens_map[special_token] = f"{special_token}a"

        # Train new tokenizer
        new_tokenizer = tokenizer.train_new_from_iterator(
            SMALL_TRAINING_CORPUS, 100, special_tokens_map=special_tokens_map
        )

        # Check the changes
        for token in special_tokens_list:
            # Get the private one to avoid unnecessary warnings.
            if getattr(tokenizer, token) is None:
                continue
            special_token = getattr(tokenizer, token)
            if special_token in special_tokens_map:
                new_special_token = getattr(new_tokenizer, token)
                self.assertEqual(special_tokens_map[special_token], new_special_token)

                new_id = new_tokenizer.get_vocab()[new_special_token]
                self.assertEqual(getattr(new_tokenizer, f"{token}_id"), new_id)

        # Check if the AddedToken / string format has been kept
        for special_token in tokenizer.all_special_tokens_extended:
            if isinstance(special_token, AddedToken) and special_token.content not in special_tokens_map:
                # The special token must appear identically in the list of the new tokenizer.
                self.assertTrue(
                    special_token in new_tokenizer.all_special_tokens_extended,
                    f"'{special_token}' should be in {new_tokenizer.all_special_tokens_extended}",
                )
            elif isinstance(special_token, AddedToken):
                # The special token must appear in the list of the new tokenizer as an object of type AddedToken with
                # the same parameters as the old AddedToken except the content that the user has requested to change.
                special_token_str = special_token.content
                new_special_token_str = special_tokens_map[special_token_str]

                find = False
                for candidate in new_tokenizer.all_special_tokens_extended:
                    if (
                        isinstance(candidate, AddedToken)
                        and candidate.content == new_special_token_str
                        and candidate.lstrip == special_token.lstrip
                        and candidate.rstrip == special_token.rstrip
                        and candidate.normalized == special_token.normalized
                        and candidate.single_word == special_token.single_word
                    ):
                        find = True
                        break
                special_token.content = new_special_token_str
                self.assertTrue(
                    find,
                    f"'{special_token.__repr__()}' should appear as an `AddedToken` in the all_special_tokens_extended = "
                    f"{[k for k in new_tokenizer.all_special_tokens_extended if str(k) == new_special_token_str]} but it is missing"
                    ", this means that the new tokenizers did not keep the `rstrip`, `lstrip`, `normalized` etc attributes.",
                )
            elif special_token not in special_tokens_map:
                # The special token must appear identically in the list of the new tokenizer.
                self.assertTrue(
                    special_token in new_tokenizer.all_special_tokens_extended,
                    f"'{special_token.__repr__()}' should be in {new_tokenizer.all_special_tokens_extended}",
                )

            else:
                # The special token must appear in the list of the new tokenizer as an object of type string.
                self.assertTrue(special_tokens_map[special_token] in new_tokenizer.all_special_tokens_extended)

        # Test we can use the new tokenizer with something not seen during training
        inputs = new_tokenizer(["This is the first sentence", "This sentence is different 🤗."])
        self.assertEqual(len(inputs["input_ids"]), 2)
        decoded_input = new_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        expected_result = "This is the first sentence"

        self.assertEqual(expected_result, decoded_input)

    def test_alignement_methods(self):
        # TODO: @ArthurZucker - alignment is broken
        pass

    def test_added_tokens_do_lower_case(self):
        # TODO: @ArthurZucker
        pass


@require_torch
@require_sentencepiece
@require_tokenizers
class MoshiIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint_name = "kmhf/hf-moshiko"
        cls.rust_tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
        return cls

    @require_torch
    def integration_tests(self):
        inputs = self.tokenizer(
            ["The following string should be properly encoded: Hello.", "But ird and ปี   ird   ด"],
            return_tensors="pt",
        )

        long_attention_mask = [1] * 21

        # fmt: off
        self.assertEqual(
            nested_simplify(inputs),
            {
                "input_ids": [
                    [287, 547, 2359, 457, 297, 3708, 11488, 279, 11725, 263],
                    [588, 478, 1442, 267, 260, 228, 188, 159, 228, 188, 185, 260, 260, 478, 1442, 260, 260, 260, 228, 188, 152],
                ],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], long_attention_mask],
            },
        )
        # fmt: on

    def test_fast_special_tokens(self):
        fast_tokenizer = self.rust_tokenizer

        fast_tokenizer.add_eos_token = False
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [318, 1145, 694]

        fast_tokenizer.add_eos_token = True
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [318, 1145, 694]

        self.rust_tokenizer.add_eos_token = False

    def test_simple_encode_decode(self):
        rust_tokenizer = self.rust_tokenizer

        self.assertEqual(rust_tokenizer.encode("This is a test"), [353, 275, 272, 694])
        self.assertEqual(rust_tokenizer.decode([353, 275, 272, 694], skip_special_tokens=True), "This is a test")

        # bytefallback showcase
        bytefallback_tokens = [260, 235, 152, 163, 234, 184, 191, 13340, 235, 160, 163, 236, 180, 159, 234, 156, 179]  # fmt: skip
        self.assertEqual(rust_tokenizer.encode("生活的真谛是"), bytefallback_tokens)
        self.assertEqual(
            rust_tokenizer.decode(bytefallback_tokens, skip_special_tokens=True),
            "生活的真谛是",
        )

        # Inner spaces showcase
        self.assertEqual(rust_tokenizer.encode("Hi  Hello"), [2769, 260, 11725])
        self.assertEqual(rust_tokenizer.decode([2769, 260, 11725], skip_special_tokens=True), "Hi  Hello")

        self.assertEqual(rust_tokenizer.encode("Hi   Hello"), [2769, 260, 260, 11725])
        self.assertEqual(rust_tokenizer.decode([2769, 260, 260, 11725], skip_special_tokens=True), "Hi   Hello")

        # TODO: @ArthurZucker
        # self.assertEqual(rust_tokenizer.encode(""), [])

        # self.assertEqual(rust_tokenizer.encode(" "), [260, 260])

        # self.assertEqual(rust_tokenizer.encode("  "), [260, 260, 260])

        # self.assertEqual(rust_tokenizer.encode(" Hello"), [260, 11725])

        # self.assertEqual(rust_tokenizer.encode("<s>"), [607, 266, 578])

    def test_no_differences_decode(self):
        rust_tokenizer = self.rust_tokenizer

        self.assertEqual(rust_tokenizer.decode([869]), "levels")

        self.assertEqual(rust_tokenizer.decode([30112, 869]), "unanswered levels")


@require_sentencepiece
@require_tokenizers
class CommonSpmIntegrationTests(unittest.TestCase):
    """
    A class that regroups important test to make sure that we properly handle the special tokens.
    """

    def test_edge_case_tabulation(self):
        fast_tokenizer = AutoTokenizer.from_pretrained("kmhf/hf-moshiko")
        input_text = "Hey<eos>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61"
        EXPECTED_IDS = [11510, 934, 4451, 266, 578, 263, 260, 13, 13, 260, 14, 14, 5209, 260, 260, 1202, 260, 527, 1322, 244, 163, 156, 140, 260, 260, 244, 163, 168, 155, 430, 1047, 261, 260, 265, 270, 278, 281, 260, 265, 280, 260, 280, 261, 285, 265]  # fmt: skip
        EXPECTED_TOKENS = ['▁Hey', '<', 'eo', 's', '>', '.', '▁', '<0x09>', '<0x09>', '▁', '<0x0A>', '<0x0A>', 'you', '▁', '▁', 'é', '▁', '▁@', '#', '<0xF0>', '<0x9F>', '<0x98>', '<0x88>', '▁', '▁', '<0xF0>', '<0x9F>', '<0xA4>', '<0x97>', '!', '▁▁▁▁▁▁▁', ',', '▁', '1', '2', '3', '4', '▁', '1', '5', '▁', '5', ',', '6', '1']  # fmt: skip

        tokens = fast_tokenizer.tokenize(input_text)
        with self.subTest("test fast edge case fast"):
            self.assertEqual(tokens, EXPECTED_TOKENS)

        input_ids = fast_tokenizer.encode(input_text)
        with self.subTest("test fast edge case fast"):
            self.assertEqual(input_ids, EXPECTED_IDS)

        text = fast_tokenizer.decode(EXPECTED_IDS)
        with self.subTest("test fast edge case fast"):
            self.assertEqual(text, "Hey<eos>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61")

        input_text = "\t\t\t\t \n\n61"
        EXPECTED_IDS = [260, 13, 13, 13, 13, 260, 14, 14, 285, 265]
        EXPECTED_TOKENS = ["▁", "<0x09>", "<0x09>", "<0x09>", "<0x09>", "▁", "<0x0A>", "<0x0A>", "6", "1"]

        tokens = fast_tokenizer.tokenize(input_text)
        with self.subTest("test fast edge case fast"):
            self.assertEqual(tokens, EXPECTED_TOKENS)

        input_ids = fast_tokenizer.encode(input_text)
        with self.subTest("test fast edge case fast"):
            self.assertEqual(input_ids, EXPECTED_IDS)

        text = fast_tokenizer.decode(EXPECTED_IDS)
        with self.subTest("test fast edge case fast"):
            self.assertEqual(text, "\t\t\t\t \n\n61")
