# coding=utf-8
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


import inspect
import re
import shutil
import tempfile
import unittest
from typing import List, Tuple

from transformers import PreTrainedTokenizerFast
from transformers.models.character_bert.tokenization_character_bert import (
    BasicTokenizer,
    CharacterBertTokenizer,
    CharacterMapper,
    _is_control,
    _is_punctuation,
    _is_whitespace,
)
from transformers.testing_utils import require_tokenizers
from transformers.utils import logging

from .test_tokenization_common import TokenizerTesterMixin, filter_non_english


logger = logging.get_logger(__name__)


@require_tokenizers
class CharacterBertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = CharacterBertTokenizer
    rust_tokenizer_class = None
    test_rust_tokenizer = False
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english

    def setUp(self):
        super().setUp()
        tokenizer = self.tokenizer_class.from_pretrained("helboukkouri/character-bert")
        tokenizer.save_pretrained(self.tmpdirname)
        logger.info(
            "Loaded pre-trained CharacterBertTokenizer " f"& saved it in: {self.tmpdirname}",
        )

    def get_clean_sequence(
        self, tokenizer, with_prefix_space=False, max_length=20, min_length=5
    ) -> Tuple[str, List[List[int]]]:
        # All the "words" that are represented as a single character (utf-8 code)
        toks = [(i + 1, chr(i)) for i in range(128)]

        toks = list(filter(lambda t: re.match(r"^[ a-zA-Z]+$", t[1]), toks))
        toks = list(
            filter(lambda t: (t[1] != " ") and (t[0] == tokenizer.encode(t[1], add_special_tokens=False)[0][1]), toks)
        )
        if max_length is not None and len(toks) > max_length:
            toks = toks[:max_length]
        if min_length is not None and len(toks) < min_length and len(toks) > 0:
            while len(toks) < min_length:
                toks = toks + toks
        toks_ids = [[i + 1 for i in tokenizer._mapper._make_char_id_sequence(t[0] - 1)] for t in toks]

        # Ensure consistency
        output_txt = tokenizer.decode(toks_ids, clean_up_tokenization_spaces=False)
        if " " not in output_txt and len(toks_ids) > 1:
            output_txt = (
                tokenizer.decode([toks_ids[0]], clean_up_tokenization_spaces=False)
                + " "
                + tokenizer.decode(toks_ids[1:], clean_up_tokenization_spaces=False)
            )
        if with_prefix_space:
            output_txt = " " + output_txt
        output_ids = tokenizer.encode(output_txt, add_special_tokens=False)
        return output_txt, output_ids

    @unittest.skip("Adding special tokens is not supported.")
    def test_add_special_tokens(self):
        pass

    @unittest.skip("Adding tokens is not supported (character-level model).")
    def test_add_tokens(self):
        pass

    @unittest.skip("Adding tokens is not supported (character-level model).")
    def test_add_tokens_tokenizer(self):
        pass

    @unittest.skip("Adding tokens is not supported (character-level model).")
    def test_added_token_serializable(self):
        pass

    @unittest.skip("Adding tokens is not supported (character-level model).")
    def test_added_tokens_do_lower_case(self):
        pass

    @unittest.skip("Adding tokens is not supported (character-level model).")
    def test_encode_decode_with_spaces(self):
        pass

    @unittest.skip("CharacterBERT does not use a token/wordpiece vocabulary.")
    def test_get_vocab(self):
        pass

    def test_conversion_reversible(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                character_vocabulary = [(i + 1, chr(i)) for i in range(128)]
                for (i, character) in character_vocabulary:
                    if character == tokenizer.unk_token:
                        continue
                    character_ids = tokenizer.convert_tokens_to_ids(character)
                    self.assertEqual(character_ids[1], i)
                    self.assertEqual(tokenizer.convert_ids_to_tokens(character_ids), character)

    def test_encode_decode(self):
        text = "[CLS] Test this : 您 [SEP]"
        tokenizer = self.get_tokenizer(do_lower_case=False)
        self.assertEqual(text, tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)))

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
                before_vocab = tokenizer.get_mlm_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                after_vocab = after_tokenizer.get_mlm_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertDictEqual(before_vocab, after_vocab)

                shutil.rmtree(tmpdirname)

    def test_tokenizer_slow_store_full_signature(self):
        signature = inspect.signature(self.tokenizer_class.__init__)
        tokenizer = self.get_tokenizer()

        for parameter_name, parameter in signature.parameters.items():
            if (parameter_name != "mlm_vocab_file") and (parameter.default != inspect.Parameter.empty):
                self.assertIn(parameter_name, tokenizer.init_kwargs)

    def test_maximum_encoding_length_single_input(self):
        tokenizers = self.get_tokenizers(do_lower_case=False, model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                seq_0, ids = self.get_clean_sequence(tokenizer, max_length=20)

                sequence = tokenizer.encode(seq_0, add_special_tokens=False)
                total_length = len(sequence)

                assert total_length > 4, "Issue with the testing sequence, please update it it's too short"

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_1 = seq_0 * model_max_length

                sequence1 = tokenizer(seq_1, add_special_tokens=False)
                total_length1 = len(sequence1["input_ids"])
                assert (
                    total_length1 > model_max_length
                ), "Issue with the testing sequence, please update it it's too short"

                # Simple
                padding_strategies = [False, True, "longest"]
                for padding_state in padding_strategies:
                    with self.subTest(f"Padding: {padding_state}"):
                        for truncation_state in [True, "longest_first", "only_first"]:
                            with self.subTest(f"Truncation: {truncation_state}"):
                                output = tokenizer(seq_1, padding=padding_state, truncation=truncation_state)
                                self.assertEqual(len(output["input_ids"]), model_max_length)

                                output = tokenizer([seq_1], padding=padding_state, truncation=truncation_state)
                                self.assertEqual(len(output["input_ids"][0]), model_max_length)

                        # Simple with no truncation
                        # Reset warnings
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer(seq_1, padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer([seq_1], padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"][0]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                # Overflowing tokens
                stride = 2
                information = tokenizer(
                    seq_0,
                    max_length=total_length - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="longest_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )

                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information["input_ids"][0]
                    overflowing_tokens = information["input_ids"][1]
                    self.assertEqual(len(information["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), total_length - 2)
                    self.assertEqual(truncated_sequence, sequence[:-2])

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, sequence[-(2 + stride) :])
                else:
                    truncated_sequence = information["input_ids"]
                    overflowing_tokens = information["overflowing_tokens"]

                    self.assertEqual(len(truncated_sequence), total_length - 2)
                    self.assertEqual(truncated_sequence, sequence[:-2])

                    self.assertEqual(len(overflowing_tokens), 2 + stride)

    def test_maximum_encoding_length_pair_input(self):
        tokenizers = self.get_tokenizers(do_lower_case=False, model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Build a sequence from our model's vocabulary
                stride = 2
                seq_0, ids = self.get_clean_sequence(tokenizer, max_length=20)
                if len(ids) <= 2 + stride:
                    seq_0 = (seq_0 + " ") * (2 + stride)
                    ids = None

                seq0_tokens = tokenizer.encode(seq_0, add_special_tokens=False)
                assert len(seq0_tokens) > 2 + stride

                seq_1 = "This is another sentence to be encoded."
                seq1_tokens = tokenizer.encode(seq_1, add_special_tokens=False)
                if abs(len(seq0_tokens) - len(seq1_tokens)) <= 2:
                    seq1_tokens = seq1_tokens + seq1_tokens
                    seq_1 = tokenizer.decode(seq1_tokens, clean_up_tokenization_spaces=False)
                seq1_tokens = tokenizer.encode(seq_1, add_special_tokens=False)

                assert len(seq1_tokens) > 2 + stride

                smallest = seq1_tokens if len(seq0_tokens) > len(seq1_tokens) else seq0_tokens

                # We are not using the special tokens - a bit too hard to test all the tokenizers with this
                # TODO try this again later
                sequence = tokenizer.encode(seq_0, seq_1, add_special_tokens=False)  # , add_prefix_space=False)

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_2 = seq_0 * model_max_length
                assert len(seq_2) > model_max_length

                sequence1 = tokenizer(seq_1, add_special_tokens=False)
                total_length1 = len(sequence1["input_ids"])
                sequence2 = tokenizer(seq_2, seq_1, add_special_tokens=False)
                total_length2 = len(sequence2["input_ids"])
                assert total_length1 < model_max_length - 10, "Issue with the testing sequence, please update it."
                assert total_length2 > model_max_length, "Issue with the testing sequence, please update it."

                # Simple
                padding_strategies = [False, True, "longest"]
                for padding_state in padding_strategies:
                    with self.subTest(f"{tokenizer.__class__.__name__} Padding: {padding_state}"):
                        for truncation_state in [True, "longest_first", "only_first"]:
                            with self.subTest(f"{tokenizer.__class__.__name__} Truncation: {truncation_state}"):
                                output = tokenizer(seq_2, seq_1, padding=padding_state, truncation=truncation_state)
                                self.assertEqual(len(output["input_ids"]), model_max_length)

                                output = tokenizer(
                                    [seq_2], [seq_1], padding=padding_state, truncation=truncation_state
                                )
                                self.assertEqual(len(output["input_ids"][0]), model_max_length)

                        # Simple
                        output = tokenizer(seq_1, seq_2, padding=padding_state, truncation="only_second")
                        self.assertEqual(len(output["input_ids"]), model_max_length)

                        output = tokenizer([seq_1], [seq_2], padding=padding_state, truncation="only_second")
                        self.assertEqual(len(output["input_ids"][0]), model_max_length)

                        # Simple with no truncation
                        # Reset warnings
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer(seq_1, seq_2, padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer([seq_1], [seq_2], padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"][0]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                truncated_first_sequence = tokenizer.encode(seq_0, add_special_tokens=False)[:-2] + tokenizer.encode(
                    seq_1, add_special_tokens=False
                )
                truncated_second_sequence = (
                    tokenizer.encode(seq_0, add_special_tokens=False)
                    + tokenizer.encode(seq_1, add_special_tokens=False)[:-2]
                )
                truncated_longest_sequence = (
                    truncated_first_sequence if len(seq0_tokens) > len(seq1_tokens) else truncated_second_sequence
                )

                overflow_first_sequence = tokenizer.encode(seq_0, add_special_tokens=False)[
                    -(2 + stride) :
                ] + tokenizer.encode(seq_1, add_special_tokens=False)
                overflow_second_sequence = (
                    tokenizer.encode(seq_0, add_special_tokens=False)
                    + tokenizer.encode(seq_1, add_special_tokens=False)[-(2 + stride) :]
                )
                overflow_longest_sequence = (
                    overflow_first_sequence if len(seq0_tokens) > len(seq1_tokens) else overflow_second_sequence
                )

                information = tokenizer.encode_plus(
                    seq_0,
                    seq_1,
                    max_length=len(sequence) - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="longest_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information["input_ids"][0]
                    overflowing_tokens = information["input_ids"][1]
                    self.assertEqual(len(information["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_longest_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(smallest))
                    self.assertEqual(overflowing_tokens, overflow_longest_sequence)
                else:
                    truncated_sequence = information["input_ids"]
                    overflowing_tokens = information["overflowing_tokens"]

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_longest_sequence)

                    self.assertEqual(
                        len(overflowing_tokens), 2 + stride
                    )  # No overflowing tokens when using 'longest' in python tokenizers

                information = tokenizer.encode_plus(
                    seq_0,
                    seq_1,
                    max_length=len(sequence) - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation=True,
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information["input_ids"][0]
                    overflowing_tokens = information["input_ids"][1]
                    self.assertEqual(len(information["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_longest_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(smallest))
                    self.assertEqual(overflowing_tokens, overflow_longest_sequence)
                else:
                    truncated_sequence = information["input_ids"]
                    overflowing_tokens = information["overflowing_tokens"]

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_longest_sequence)

                    self.assertEqual(
                        len(overflowing_tokens), 2 + stride
                    )  # No overflowing tokens when using 'longest' in python tokenizers

                information_first_truncated = tokenizer.encode_plus(
                    seq_0,
                    seq_1,
                    max_length=len(sequence) - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="only_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information_first_truncated["input_ids"][0]
                    overflowing_tokens = information_first_truncated["input_ids"][1]
                    self.assertEqual(len(information_first_truncated["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_first_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(seq1_tokens))
                    self.assertEqual(overflowing_tokens, overflow_first_sequence)
                else:
                    truncated_sequence = information_first_truncated["input_ids"]
                    overflowing_tokens = information_first_truncated["overflowing_tokens"]

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_first_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, seq0_tokens[-(2 + stride) :])

                information_second_truncated = tokenizer.encode_plus(
                    seq_0,
                    seq_1,
                    max_length=len(sequence) - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="only_second",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information_second_truncated["input_ids"][0]
                    overflowing_tokens = information_second_truncated["input_ids"][1]
                    self.assertEqual(len(information_second_truncated["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_second_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(seq0_tokens))
                    self.assertEqual(overflowing_tokens, overflow_second_sequence)
                else:
                    truncated_sequence = information_second_truncated["input_ids"]
                    overflowing_tokens = information_second_truncated["overflowing_tokens"]

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_second_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, seq1_tokens[-(2 + stride) :])

    def test_full_tokenizer(self):
        sequence = "Hi!"

        # Test default (lowercase)
        tokenizer = self.tokenizer_class()
        expected_tokens = ["hi", "!"]  # Same as BasicTokenizer
        expected_ids = [
            # Each word is represented as a sequence of characters
            # beginning with special beginning of word character,
            # ending with a special end of word character
            # then followed with as many padding characters as needed
            # to fill a `max_word_length` (default=50) sequence
            [
                CharacterMapper.beginning_of_word_character + 1,
                105,
                106,
                CharacterMapper.end_of_word_character + 1,
            ]
            + [CharacterMapper.padding_character + 1] * (tokenizer.max_word_length - 4),
            [
                CharacterMapper.beginning_of_word_character + 1,
                34,
                CharacterMapper.end_of_word_character + 1,
            ]
            + [CharacterMapper.padding_character + 1] * (tokenizer.max_word_length - 3),
        ]
        tokens = tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, expected_tokens)
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), expected_ids)
        ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, expected_ids)

        # Test UpperCase
        tokenizer = self.tokenizer_class.from_pretrained("helboukkouri/character-bert", do_lower_case=False)
        expected_tokens = ["Hi", "!"]  # Same as BasicTokenizer
        expected_ids = [
            # Each word is represented as a sequence of characters
            # beginning with special beginning of word character,
            # ending with a special end of word character
            # then followed with as many padding characters as needed
            # to fill a `max_word_length` (default=50) sequence
            [
                CharacterMapper.beginning_of_word_character + 1,
                73,
                106,
                CharacterMapper.end_of_word_character + 1,
            ]
            + [CharacterMapper.padding_character + 1] * (tokenizer.max_word_length - 4),
            [
                CharacterMapper.beginning_of_word_character + 1,
                34,
                CharacterMapper.end_of_word_character + 1,
            ]
            + [CharacterMapper.padding_character + 1] * (tokenizer.max_word_length - 3),
        ]
        tokens = tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, expected_tokens)
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), expected_ids)
        ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, expected_ids)

    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("helboukkouri/character-bert")
        CLS = [CharacterMapper.beginning_of_word_character + 1, 257, CharacterMapper.end_of_word_character + 1] + [
            CharacterMapper.padding_character + 1
        ] * (tokenizer.max_word_length - 3)
        SEP = [CharacterMapper.beginning_of_word_character + 1, 258, CharacterMapper.end_of_word_character + 1] + [
            CharacterMapper.padding_character + 1
        ] * (tokenizer.max_word_length - 3)

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [CLS] + text + [SEP]
        assert encoded_pair == [CLS] + text + [SEP] + text_2 + [SEP]

    def test_basic_tokenizer_chinese(self):
        tokenizer = BasicTokenizer()

        self.assertListEqual(tokenizer.tokenize("ah\u535A\u63A8zz"), ["ah", "\u535A", "\u63A8", "zz"])

    def test_basic_tokenizer_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "), ["hello", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hällo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["h\u00E9llo"])

    def test_basic_tokenizer_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hallo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_default(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hallo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "), ["HeLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_no_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["HäLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_no_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["HaLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_respects_never_split_tokens(self):
        tokenizer = BasicTokenizer(do_lower_case=False, never_split=["[UNK]"])

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU? [UNK]"), ["HeLLo", "!", "how", "Are", "yoU", "?", "[UNK]"]
        )

    def test_is_whitespace(self):
        self.assertTrue(_is_whitespace(" "))
        self.assertTrue(_is_whitespace("\t"))
        self.assertTrue(_is_whitespace("\r"))
        self.assertTrue(_is_whitespace("\n"))
        self.assertTrue(_is_whitespace("\u00A0"))

        self.assertFalse(_is_whitespace("A"))
        self.assertFalse(_is_whitespace("-"))

    def test_is_control(self):
        self.assertTrue(_is_control("\u0005"))

        self.assertFalse(_is_control("A"))
        self.assertFalse(_is_control(" "))
        self.assertFalse(_is_control("\t"))
        self.assertFalse(_is_control("\r"))

    def test_is_punctuation(self):
        self.assertTrue(_is_punctuation("-"))
        self.assertTrue(_is_punctuation("$"))
        self.assertTrue(_is_punctuation("`"))
        self.assertTrue(_is_punctuation("."))

        self.assertFalse(_is_punctuation("A"))
        self.assertFalse(_is_punctuation(" "))
