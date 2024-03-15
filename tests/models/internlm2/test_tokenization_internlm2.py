# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Team. All rights reserved.
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
import unittest

from transformers import AddedToken, InternLM2Tokenizer, InternLM2TokenizerFast
from transformers.testing_utils import require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin, SpecialTokensMixin, SMALL_TRAINING_CORPUS


@require_tokenizers
class InternLM2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = InternLM2Tokenizer
    rust_tokenizer_class = InternLM2TokenizerFast
    test_slow_tokenizer = True
    test_rust_tokenizer = True
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = InternLM2Tokenizer.from_pretrained(
            "internlm/internlm2-chat-7b",
            revision="5b50661e5ba16c9ded1047a51e394280b3b9bda1"
        )
        tokenizer.save_pretrained(self.tmpdirname)

    def get_text_and_tokens(self):
        text = "This is a test text for internlm2 tokenizer: How are you today?"
        tokens = [2136, 505, 395, 1420, 1614, 500, 2750, 17912, 314, 45433, 334, 2745, 657, 629, 3514, 345]

        return text, tokens

    def test_integration(self, **kwargs):
        text = "This is a test text for internlm2 tokenizer: How are you today?"
        expected = {
            "input_ids":[[1, 2136, 505, 395, 1420, 1614, 500, 2750, 17912, 314, 45433, 334, 2745, 657, 629, 3514, 345]],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        }

        self.tokenizer_integration_test_util(
            sequences=[text],
            expected_encoding=expected,
            model_name="internlm/internlm2-chat-7b",
            revision="5b50661e5ba16c9ded1047a51e394280b3b9bda1",
            padding=False,
        )

    def test_decode_one_by_one(self, **kwargs):
        # mainly for stream chat
        text, tokens = self.get_text_and_tokens()
        tokenizer_classes = [self.tokenizer_class]
        if self.test_rust_tokenizer:
            tokenizer_classes.append(self.rust_tokenizer_class)

        for tokenizer_class in tokenizer_classes:
            tokenizer = tokenizer_class.from_pretrained(
                "internlm/internlm2-chat-7b", revision="5b50661e5ba16c9ded1047a51e394280b3b9bda1"
            )
            decoded = ""
            for token in tokens:
                decoded += tokenizer.decode(token)
            self.assertEqual(decoded, text)

    def test_special_chat_tokens(self, **kwargs):
        special_chat_tokens = {
            "<|im_start|>": 92543,
            "<|im_end|>": 92542,
            "<|action_start|>": 92541,
            "<|action_end|>": 92540,
            "<|interpreter|>": 92539,
            "<|plugin|>": 92538,
        }

        tokenizer_classes = [self.tokenizer_class]
        if self.test_rust_tokenizer:
            tokenizer_classes.append(self.rust_tokenizer_class)

        for tokenizer_class in tokenizer_classes:
            tokenizer = tokenizer_class.from_pretrained(
                "internlm/internlm2-chat-7b", revision="5b50661e5ba16c9ded1047a51e394280b3b9bda1"
            )

            for word, token_id in special_chat_tokens.items():
                self.assertEqual(tokenizer.convert_tokens_to_ids(word), token_id)
                self.assertEqual(tokenizer.convert_ids_to_tokens(token_id), word)

    def test_alignement_methods(self):
        self.skipTest("word_ids are always 0.")
        super().test_alignement_methods

    def test_pretokenized_inputs(self):
        self.skipTest("Spaces will affect the encode results.")
        super().test_pretokenized_inputs()

    def test_training_new_tokenizer(self):
        # This feature only exists for fast tokenizers
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_rust_tokenizer()
        new_tokenizer = tokenizer.train_new_from_iterator(SMALL_TRAINING_CORPUS, 100)

        # Test we can use the new tokenizer with something not seen during training
        inputs = new_tokenizer(["This is the first sentence", "This sentence is different 🤗."])
        self.assertEqual(len(inputs["input_ids"]), 2)
        decoded_input = new_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        expected_result = "This is the first sentence"

        # remove normalizer to fit sp model
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
            return

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
            if getattr(tokenizer, f"_{token}") is not None:
                special_token = getattr(tokenizer, token)
                special_tokens_map[special_token] = f"{special_token}a"

        # Train new tokenizer
        new_tokenizer = tokenizer.train_new_from_iterator(
            SMALL_TRAINING_CORPUS, 100, special_tokens_map=special_tokens_map
        )

        # Check the changes
        for token in special_tokens_list:
            # Get the private one to avoid unnecessary warnings.
            if getattr(tokenizer, f"_{token}") is None:
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
                    f"{[k for k in new_tokenizer.all_special_tokens_extended if str(k)==new_special_token_str]} but it is missing"
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

        # remove normalizer to fit sp model
        self.assertEqual(expected_result, decoded_input)

    def test_added_tokens_do_lower_case(self):
        self.skipTest("Additional _ will be encoded in fast tokenizer, wait to be fixed.")
        super().test_added_tokens_do_lower_case()

    def test_special_tokens_initialization(self):
        self.skipTest("Additional _ will be encoded in fast tokenizer, wait to be fixed.")
        super().test_special_tokens_initialization()