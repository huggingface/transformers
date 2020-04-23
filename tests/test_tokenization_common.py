# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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


import os
import pickle
import shutil
import tempfile
from collections import OrderedDict
from typing import Dict, Tuple, Union

from tests.utils import require_tf, require_torch


def merge_model_tokenizer_mappings(
    model_mapping: Dict["PretrainedConfig", Union["PreTrainedModel", "TFPreTrainedModel"]],
    tokenizer_mapping: Dict["PretrainedConfig", Tuple["PreTrainedTokenizer", "PreTrainedTokenizerFast"]],
) -> Dict[
    Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
    Tuple["PretrainedConfig", Union["PreTrainedModel", "TFPreTrainedModel"]],
]:
    configurations = list(model_mapping.keys())
    model_tokenizer_mapping = OrderedDict([])

    for configuration in configurations:
        model = model_mapping[configuration]
        tokenizer = tokenizer_mapping[configuration][0]
        tokenizer_fast = tokenizer_mapping[configuration][1]

        model_tokenizer_mapping.update({tokenizer: (configuration, model)})
        if tokenizer_fast is not None:
            model_tokenizer_mapping.update({tokenizer_fast: (configuration, model)})

    return model_tokenizer_mapping


class TokenizerTesterMixin:

    tokenizer_class = None
    test_rust_tokenizer = False

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        raise NotImplementedError

    def get_rust_tokenizer(self, **kwargs):
        raise NotImplementedError

    def get_input_output_texts(self):
        raise NotImplementedError

    @staticmethod
    def convert_batch_encode_plus_format_to_encode_plus(batch_encode_plus_sequences):
        # Switch from batch_encode_plus format:   {'input_ids': [[...], [...]], ...}
        # to the concatenated encode_plus format: [{'input_ids': [...], ...}, {'input_ids': [...], ...}]
        return [
            {value: batch_encode_plus_sequences[value][i] for value in batch_encode_plus_sequences.keys()}
            for i in range(len(batch_encode_plus_sequences["input_ids"]))
        ]

    def test_tokenizers_common_properties(self):
        tokenizer = self.get_tokenizer()
        attributes_list = [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
        ]
        for attr in attributes_list:
            self.assertTrue(hasattr(tokenizer, attr))
            self.assertTrue(hasattr(tokenizer, attr + "_id"))

        self.assertTrue(hasattr(tokenizer, "additional_special_tokens"))
        self.assertTrue(hasattr(tokenizer, "additional_special_tokens_ids"))

        attributes_list = ["max_len", "init_inputs", "init_kwargs", "added_tokens_encoder", "added_tokens_decoder"]
        for attr in attributes_list:
            self.assertTrue(hasattr(tokenizer, attr))

    def test_save_and_load_tokenizer(self):
        # safety check on max_len default value so we are sure the test works
        tokenizer = self.get_tokenizer()
        self.assertNotEqual(tokenizer.max_len, 42)

        # Now let's start the test
        tokenizer = self.get_tokenizer(max_len=42)

        before_tokens = tokenizer.encode("He is very happy, UNwant\u00E9d,running", add_special_tokens=False)

        tokenizer.save_pretrained(self.tmpdirname)
        tokenizer = self.tokenizer_class.from_pretrained(self.tmpdirname)

        after_tokens = tokenizer.encode("He is very happy, UNwant\u00E9d,running", add_special_tokens=False)
        self.assertListEqual(before_tokens, after_tokens)

        self.assertEqual(tokenizer.max_len, 42)
        tokenizer = self.tokenizer_class.from_pretrained(self.tmpdirname, max_len=43)
        self.assertEqual(tokenizer.max_len, 43)

    def test_pickle_tokenizer(self):
        tokenizer = self.get_tokenizer()
        self.assertIsNotNone(tokenizer)

        text = "Munich and Berlin are nice cities"
        subwords = tokenizer.tokenize(text)

        filename = os.path.join(self.tmpdirname, "tokenizer.bin")
        with open(filename, "wb") as handle:
            pickle.dump(tokenizer, handle)

        with open(filename, "rb") as handle:
            tokenizer_new = pickle.load(handle)

        subwords_loaded = tokenizer_new.tokenize(text)

        self.assertListEqual(subwords, subwords_loaded)

    def test_added_tokens_do_lower_case(self):
        tokenizer = self.get_tokenizer(do_lower_case=True)

        special_token = tokenizer.all_special_tokens[0]

        text = special_token + " aaaaa bbbbbb low cccccccccdddddddd l " + special_token
        text2 = special_token + " AAAAA BBBBBB low CCCCCCCCCDDDDDDDD l " + special_token

        toks0 = tokenizer.tokenize(text)  # toks before adding new_toks

        new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd", "AAAAA BBBBBB", "CCCCCCCCCDDDDDDDD"]
        added = tokenizer.add_tokens(new_toks)
        self.assertEqual(added, 2)

        toks = tokenizer.tokenize(text)
        toks2 = tokenizer.tokenize(text2)

        self.assertEqual(len(toks), len(toks2))
        self.assertNotEqual(len(toks), len(toks0))  # toks0 should be longer
        self.assertListEqual(toks, toks2)

        # Check that none of the special tokens are lowercased
        sequence_with_special_tokens = "A " + " yEs ".join(tokenizer.all_special_tokens) + " B"
        tokenized_sequence = tokenizer.tokenize(sequence_with_special_tokens)

        for special_token in tokenizer.all_special_tokens:
            self.assertTrue(special_token in tokenized_sequence)

        tokenizer = self.get_tokenizer(do_lower_case=False)

        added = tokenizer.add_tokens(new_toks)
        self.assertEqual(added, 4)

        toks = tokenizer.tokenize(text)
        toks2 = tokenizer.tokenize(text2)

        self.assertEqual(len(toks), len(toks2))  # Length should still be the same
        self.assertNotEqual(len(toks), len(toks0))
        self.assertNotEqual(toks[1], toks2[1])  # But at least the first non-special tokens should differ

    def test_add_tokens_tokenizer(self):
        tokenizer = self.get_tokenizer()

        vocab_size = tokenizer.vocab_size
        all_size = len(tokenizer)

        self.assertNotEqual(vocab_size, 0)
        self.assertEqual(vocab_size, all_size)

        new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
        added_toks = tokenizer.add_tokens(new_toks)
        vocab_size_2 = tokenizer.vocab_size
        all_size_2 = len(tokenizer)

        self.assertNotEqual(vocab_size_2, 0)
        self.assertEqual(vocab_size, vocab_size_2)
        self.assertEqual(added_toks, len(new_toks))
        self.assertEqual(all_size_2, all_size + len(new_toks))

        tokens = tokenizer.encode("aaaaa bbbbbb low cccccccccdddddddd l", add_special_tokens=False)

        self.assertGreaterEqual(len(tokens), 4)
        self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

        new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||>|>>>>|>"}
        added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
        vocab_size_3 = tokenizer.vocab_size
        all_size_3 = len(tokenizer)

        self.assertNotEqual(vocab_size_3, 0)
        self.assertEqual(vocab_size, vocab_size_3)
        self.assertEqual(added_toks_2, len(new_toks_2))
        self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

        tokens = tokenizer.encode(
            ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l", add_special_tokens=False
        )

        self.assertGreaterEqual(len(tokens), 6)
        self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[0], tokens[1])
        self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[-2], tokens[-3])
        self.assertEqual(tokens[0], tokenizer.eos_token_id)
        self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    def test_add_special_tokens(self):
        tokenizer = self.get_tokenizer()
        input_text, output_text = self.get_input_output_texts()

        special_token = "[SPECIAL TOKEN]"

        tokenizer.add_special_tokens({"cls_token": special_token})
        encoded_special_token = tokenizer.encode(special_token, add_special_tokens=False)
        assert len(encoded_special_token) == 1

        text = " ".join([input_text, special_token, output_text])
        encoded = tokenizer.encode(text, add_special_tokens=False)

        input_encoded = tokenizer.encode(input_text, add_special_tokens=False)
        output_encoded = tokenizer.encode(" " + output_text, add_special_tokens=False)
        special_token_id = tokenizer.encode(special_token, add_special_tokens=False)
        assert encoded == input_encoded + special_token_id + output_encoded

        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        assert special_token not in decoded

    def test_required_methods_tokenizer(self):
        tokenizer = self.get_tokenizer()
        input_text, output_text = self.get_input_output_texts()

        tokens = tokenizer.tokenize(input_text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        ids_2 = tokenizer.encode(input_text, add_special_tokens=False)
        self.assertListEqual(ids, ids_2)

        tokens_2 = tokenizer.convert_ids_to_tokens(ids)
        text_2 = tokenizer.decode(ids)

        self.assertEqual(text_2, output_text)

        self.assertNotEqual(len(tokens_2), 0)
        self.assertIsInstance(text_2, str)

    def test_encode_decode_with_spaces(self):
        tokenizer = self.get_tokenizer()

        new_toks = ["[ABC]", "[DEF]", "GHI IHG"]
        tokenizer.add_tokens(new_toks)
        input = "[ABC] [DEF] [ABC] GHI IHG [DEF]"
        encoded = tokenizer.encode(input, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, input)

    def test_pretrained_model_lists(self):
        weights_list = list(self.tokenizer_class.max_model_input_sizes.keys())
        weights_lists_2 = []
        for file_id, map_list in self.tokenizer_class.pretrained_vocab_files_map.items():
            weights_lists_2.append(list(map_list.keys()))

        for weights_list_2 in weights_lists_2:
            self.assertListEqual(weights_list, weights_list_2)

    def test_mask_output(self):
        tokenizer = self.get_tokenizer()

        if (
            tokenizer.build_inputs_with_special_tokens.__qualname__.split(".")[0] != "PreTrainedTokenizer"
            and "token_type_ids" in tokenizer.model_input_names
        ):
            seq_0 = "Test this method."
            seq_1 = "With these inputs."
            information = tokenizer.encode_plus(seq_0, seq_1, add_special_tokens=True)
            sequences, mask = information["input_ids"], information["token_type_ids"]
            self.assertEqual(len(sequences), len(mask))

    def test_number_of_added_tokens(self):
        tokenizer = self.get_tokenizer()

        seq_0 = "Test this method."
        seq_1 = "With these inputs."

        sequences = tokenizer.encode(seq_0, seq_1, add_special_tokens=False)
        attached_sequences = tokenizer.encode(seq_0, seq_1, add_special_tokens=True, add_prefix_space=False)

        # Method is implemented (e.g. not GPT-2)
        if len(attached_sequences) != 2:
            self.assertEqual(tokenizer.num_special_tokens_to_add(pair=True), len(attached_sequences) - len(sequences))

    def test_maximum_encoding_length_single_input(self):
        tokenizer = self.get_tokenizer()

        seq_0 = "This is a sentence to be encoded."
        stride = 2

        sequence = tokenizer.encode(seq_0, add_special_tokens=False)
        num_added_tokens = tokenizer.num_special_tokens_to_add()
        total_length = len(sequence) + num_added_tokens
        information = tokenizer.encode_plus(
            seq_0,
            max_length=total_length - 2,
            add_special_tokens=True,
            stride=stride,
            return_overflowing_tokens=True,
            add_prefix_space=False,
        )

        truncated_sequence = information["input_ids"]
        overflowing_tokens = information["overflowing_tokens"]

        self.assertEqual(len(overflowing_tokens), 2 + stride)
        self.assertEqual(overflowing_tokens, sequence[-(2 + stride) :])
        self.assertEqual(len(truncated_sequence), total_length - 2)
        self.assertEqual(truncated_sequence, tokenizer.build_inputs_with_special_tokens(sequence[:-2]))

    def test_maximum_encoding_length_pair_input(self):
        tokenizer = self.get_tokenizer()

        seq_0 = "This is a sentence to be encoded."
        seq_1 = "This is another sentence to be encoded."
        stride = 2

        sequence_0_no_special_tokens = tokenizer.encode(seq_0, add_special_tokens=False)
        sequence_1_no_special_tokens = tokenizer.encode(seq_1, add_special_tokens=False)

        sequence = tokenizer.encode(seq_0, seq_1, add_special_tokens=True, add_prefix_space=False)
        truncated_second_sequence = tokenizer.build_inputs_with_special_tokens(
            tokenizer.encode(seq_0, add_special_tokens=False), tokenizer.encode(seq_1, add_special_tokens=False)[:-2],
        )

        information = tokenizer.encode_plus(
            seq_0,
            seq_1,
            max_length=len(sequence) - 2,
            add_special_tokens=True,
            stride=stride,
            truncation_strategy="only_second",
            return_overflowing_tokens=True,
            add_prefix_space=False,
        )
        information_first_truncated = tokenizer.encode_plus(
            seq_0,
            seq_1,
            max_length=len(sequence) - 2,
            add_special_tokens=True,
            stride=stride,
            truncation_strategy="only_first",
            return_overflowing_tokens=True,
            add_prefix_space=False,
        )

        truncated_sequence = information["input_ids"]
        overflowing_tokens = information["overflowing_tokens"]
        overflowing_tokens_first_truncated = information_first_truncated["overflowing_tokens"]

        self.assertEqual(len(overflowing_tokens), 2 + stride)
        self.assertEqual(overflowing_tokens, sequence_1_no_special_tokens[-(2 + stride) :])
        self.assertEqual(overflowing_tokens_first_truncated, sequence_0_no_special_tokens[-(2 + stride) :])
        self.assertEqual(len(truncated_sequence), len(sequence) - 2)
        self.assertEqual(truncated_sequence, truncated_second_sequence)

    def test_encode_input_type(self):
        tokenizer = self.get_tokenizer()

        sequence = "Let's encode this sequence"

        tokens = tokenizer.tokenize(sequence)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        formatted_input = tokenizer.encode(sequence, add_special_tokens=True, add_prefix_space=False)

        self.assertEqual(tokenizer.encode(tokens, add_special_tokens=True), formatted_input)
        self.assertEqual(tokenizer.encode(input_ids, add_special_tokens=True), formatted_input)

    def test_swap_special_token(self):
        tokenizer = self.get_tokenizer()

        mask = "<mask>"
        sequence = "Encode this sequence"
        sequence_masked_0 = "Encode <mask> sequence"
        sequence_masked_1 = "<mask> this sequence"

        # Add tokens so that masked token isn't split
        tokenizer.add_tokens(sequence.split())
        tokenizer.add_special_tokens({"mask_token": mask})
        mask_ind = tokenizer.convert_tokens_to_ids(mask)
        encoded = tokenizer.encode(sequence, add_special_tokens=False)

        # Test first masked sequence
        encoded_masked = tokenizer.encode(sequence_masked_0, add_special_tokens=False)
        mask_loc = encoded_masked.index(mask_ind)
        encoded_masked[mask_loc] = encoded[mask_loc]

        self.assertEqual(encoded_masked, encoded)

        # Test second masked sequence
        encoded_masked = tokenizer.encode(sequence_masked_1, add_special_tokens=False)
        mask_loc = encoded_masked.index(mask_ind)
        encoded_masked[mask_loc] = encoded[mask_loc]

        self.assertEqual(encoded_masked, encoded)

    def test_special_tokens_mask(self):
        tokenizer = self.get_tokenizer()

        sequence_0 = "Encode this."
        sequence_1 = "This one too please."

        # Testing single inputs
        encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False)
        encoded_sequence_dict = tokenizer.encode_plus(
            sequence_0, add_special_tokens=True, return_special_tokens_mask=True, add_prefix_space=False
        )
        encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
        special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
        self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

        filtered_sequence = [
            (x if not special_tokens_mask[i] else None) for i, x in enumerate(encoded_sequence_w_special)
        ]
        filtered_sequence = [x for x in filtered_sequence if x is not None]
        self.assertEqual(encoded_sequence, filtered_sequence)

        # Testing inputs pairs
        encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False)
        encoded_sequence += tokenizer.encode(sequence_1, add_special_tokens=False)
        encoded_sequence_dict = tokenizer.encode_plus(
            sequence_0, sequence_1, add_special_tokens=True, return_special_tokens_mask=True, add_prefix_space=False
        )
        encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
        special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
        self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

        filtered_sequence = [
            (x if not special_tokens_mask[i] else None) for i, x in enumerate(encoded_sequence_w_special)
        ]
        filtered_sequence = [x for x in filtered_sequence if x is not None]
        self.assertEqual(encoded_sequence, filtered_sequence)

        # Testing with already existing special tokens
        if tokenizer.cls_token_id == tokenizer.unk_token_id and tokenizer.cls_token_id == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({"cls_token": "</s>", "sep_token": "<s>"})
        encoded_sequence_dict = tokenizer.encode_plus(
            sequence_0, add_special_tokens=True, return_special_tokens_mask=True
        )
        encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
        special_tokens_mask_orig = encoded_sequence_dict["special_tokens_mask"]
        special_tokens_mask = tokenizer.get_special_tokens_mask(
            encoded_sequence_w_special, already_has_special_tokens=True
        )
        self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))
        self.assertEqual(special_tokens_mask_orig, special_tokens_mask)

    def test_padding_to_max_length(self):
        tokenizer = self.get_tokenizer()

        sequence = "Sequence"
        padding_size = 10

        # check correct behaviour if no pad_token_id exists and add it eventually
        self._check_no_pad_token_padding(tokenizer, sequence)

        padding_idx = tokenizer.pad_token_id

        # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
        tokenizer.padding_side = "right"
        encoded_sequence = tokenizer.encode(sequence)
        sequence_length = len(encoded_sequence)
        padded_sequence = tokenizer.encode(sequence, max_length=sequence_length + padding_size, pad_to_max_length=True)
        padded_sequence_length = len(padded_sequence)
        assert sequence_length + padding_size == padded_sequence_length
        assert encoded_sequence + [padding_idx] * padding_size == padded_sequence

        # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
        tokenizer.padding_side = "left"
        encoded_sequence = tokenizer.encode(sequence)
        sequence_length = len(encoded_sequence)
        padded_sequence = tokenizer.encode(sequence, max_length=sequence_length + padding_size, pad_to_max_length=True)
        padded_sequence_length = len(padded_sequence)
        assert sequence_length + padding_size == padded_sequence_length
        assert [padding_idx] * padding_size + encoded_sequence == padded_sequence

        # RIGHT & LEFT PADDING - Check that nothing is done when a maximum length is not specified
        encoded_sequence = tokenizer.encode(sequence)
        sequence_length = len(encoded_sequence)

        tokenizer.padding_side = "right"
        padded_sequence_right = tokenizer.encode(sequence, pad_to_max_length=True)
        padded_sequence_right_length = len(padded_sequence_right)

        tokenizer.padding_side = "left"
        padded_sequence_left = tokenizer.encode(sequence, pad_to_max_length=True)
        padded_sequence_left_length = len(padded_sequence_left)

        assert sequence_length == padded_sequence_right_length
        assert encoded_sequence == padded_sequence_right
        assert sequence_length == padded_sequence_left_length
        assert encoded_sequence == padded_sequence_left

    def test_encode_plus_with_padding(self):
        tokenizer = self.get_tokenizer()

        sequence = "Sequence"

        # check correct behaviour if no pad_token_id exists and add it eventually
        self._check_no_pad_token_padding(tokenizer, sequence)

        padding_size = 10
        padding_idx = tokenizer.pad_token_id
        token_type_padding_idx = tokenizer.pad_token_type_id

        encoded_sequence = tokenizer.encode_plus(sequence, return_special_tokens_mask=True)
        input_ids = encoded_sequence["input_ids"]
        special_tokens_mask = encoded_sequence["special_tokens_mask"]
        sequence_length = len(input_ids)

        # Test right padding
        tokenizer.padding_side = "right"

        right_padded_sequence = tokenizer.encode_plus(
            sequence,
            max_length=sequence_length + padding_size,
            pad_to_max_length=True,
            return_special_tokens_mask=True,
        )
        right_padded_input_ids = right_padded_sequence["input_ids"]

        right_padded_special_tokens_mask = right_padded_sequence["special_tokens_mask"]
        right_padded_sequence_length = len(right_padded_input_ids)

        assert sequence_length + padding_size == right_padded_sequence_length
        assert input_ids + [padding_idx] * padding_size == right_padded_input_ids
        assert special_tokens_mask + [1] * padding_size == right_padded_special_tokens_mask

        # Test left padding
        tokenizer.padding_side = "left"
        left_padded_sequence = tokenizer.encode_plus(
            sequence,
            max_length=sequence_length + padding_size,
            pad_to_max_length=True,
            return_special_tokens_mask=True,
        )
        left_padded_input_ids = left_padded_sequence["input_ids"]
        left_padded_special_tokens_mask = left_padded_sequence["special_tokens_mask"]
        left_padded_sequence_length = len(left_padded_input_ids)

        assert sequence_length + padding_size == left_padded_sequence_length
        assert [padding_idx] * padding_size + input_ids == left_padded_input_ids
        assert [1] * padding_size + special_tokens_mask == left_padded_special_tokens_mask

        if "token_type_ids" in tokenizer.model_input_names:
            token_type_ids = encoded_sequence["token_type_ids"]
            left_padded_token_type_ids = left_padded_sequence["token_type_ids"]
            right_padded_token_type_ids = right_padded_sequence["token_type_ids"]

            assert token_type_ids + [token_type_padding_idx] * padding_size == right_padded_token_type_ids
            assert [token_type_padding_idx] * padding_size + token_type_ids == left_padded_token_type_ids

        if "attention_mask" in tokenizer.model_input_names:
            attention_mask = encoded_sequence["attention_mask"]
            right_padded_attention_mask = right_padded_sequence["attention_mask"]
            left_padded_attention_mask = left_padded_sequence["attention_mask"]

            assert attention_mask + [0] * padding_size == right_padded_attention_mask
            assert [0] * padding_size + attention_mask == left_padded_attention_mask

    def test_separate_tokenizers(self):
        # This tests that tokenizers don't impact others. Unfortunately the case where it fails is when
        # we're loading an S3 configuration from a pre-trained identifier, and we have no way of testing those today.

        tokenizer = self.get_tokenizer(random_argument=True)
        assert tokenizer.init_kwargs["random_argument"] is True
        new_tokenizer = self.get_tokenizer(random_argument=False)
        assert tokenizer.init_kwargs["random_argument"] is True
        assert new_tokenizer.init_kwargs["random_argument"] is False

    def test_get_vocab(self):
        tokenizer = self.get_tokenizer()
        vocab = tokenizer.get_vocab()

        self.assertIsInstance(vocab, dict)
        self.assertEqual(len(vocab), len(tokenizer))

        for word, ind in vocab.items():
            self.assertEqual(tokenizer.convert_tokens_to_ids(word), ind)
            self.assertEqual(tokenizer.convert_ids_to_tokens(ind), word)

        tokenizer.add_tokens(["asdfasdfasdfasdf"])
        vocab = tokenizer.get_vocab()
        self.assertIsInstance(vocab, dict)
        self.assertEqual(len(vocab), len(tokenizer))

        for word, ind in vocab.items():
            self.assertEqual(tokenizer.convert_tokens_to_ids(word), ind)
            self.assertEqual(tokenizer.convert_ids_to_tokens(ind), word)

    def test_batch_encode_plus_batch_sequence_length(self):
        # Tests that all encoded values have the correct size
        tokenizer = self.get_tokenizer()
        sequences = [
            "Testing batch encode plus",
            "Testing batch encode plus with different sequence lengths",
            "Testing batch encode plus with different sequence lengths correctly pads",
        ]

        encoded_sequences = [tokenizer.encode_plus(sequence, pad_to_max_length=False) for sequence in sequences]
        encoded_sequences_batch = tokenizer.batch_encode_plus(sequences)
        self.assertListEqual(
            encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
        )

        maximum_length = len(max([encoded_sequence["input_ids"] for encoded_sequence in encoded_sequences], key=len))

        # check correct behaviour if no pad_token_id exists and add it eventually
        self._check_no_pad_token_padding(tokenizer, sequences)

        encoded_sequences_padded = [
            tokenizer.encode_plus(sequence, pad_to_max_length=True, max_length=maximum_length)
            for sequence in sequences
        ]

        encoded_sequences_batch_padded = tokenizer.batch_encode_plus(sequences, pad_to_max_length=True)
        self.assertListEqual(
            encoded_sequences_padded,
            self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch_padded),
        )

    def test_batch_encode_plus_padding(self):
        # Test that padded sequences are equivalent between batch_encode_plus and encode_plus

        # Right padding tests
        tokenizer = self.get_tokenizer()
        sequences = [
            "Testing batch encode plus",
            "Testing batch encode plus with different sequence lengths",
            "Testing batch encode plus with different sequence lengths correctly pads",
        ]

        max_length = 100

        # check correct behaviour if no pad_token_id exists and add it eventually
        self._check_no_pad_token_padding(tokenizer, sequences)

        encoded_sequences = [
            tokenizer.encode_plus(sequence, pad_to_max_length=True, max_length=max_length) for sequence in sequences
        ]
        encoded_sequences_batch = tokenizer.batch_encode_plus(sequences, pad_to_max_length=True, max_length=max_length)
        self.assertListEqual(
            encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
        )

        # Left padding tests
        tokenizer = self.get_tokenizer()

        tokenizer.padding_side = "left"
        sequences = [
            "Testing batch encode plus",
            "Testing batch encode plus with different sequence lengths",
            "Testing batch encode plus with different sequence lengths correctly pads",
        ]

        max_length = 100

        # check correct behaviour if no pad_token_id exists and add it eventually
        self._check_no_pad_token_padding(tokenizer, sequences)

        encoded_sequences = [
            tokenizer.encode_plus(sequence, pad_to_max_length=True, max_length=max_length) for sequence in sequences
        ]
        encoded_sequences_batch = tokenizer.batch_encode_plus(sequences, pad_to_max_length=True, max_length=max_length)
        self.assertListEqual(
            encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
        )

    @require_torch
    @require_tf
    def test_batch_encode_plus_tensors(self):
        tokenizer = self.get_tokenizer()
        sequences = [
            "Testing batch encode plus",
            "Testing batch encode plus with different sequence lengths",
            "Testing batch encode plus with different sequence lengths correctly pads",
        ]

        # A Tensor cannot be build by sequences which are not the same size
        self.assertRaises(ValueError, tokenizer.batch_encode_plus, sequences, return_tensors="pt")
        self.assertRaises(ValueError, tokenizer.batch_encode_plus, sequences, return_tensors="tf")

        if tokenizer.pad_token_id is None:
            self.assertRaises(
                ValueError, tokenizer.batch_encode_plus, sequences, pad_to_max_length=True, return_tensors="pt"
            )
            self.assertRaises(
                ValueError, tokenizer.batch_encode_plus, sequences, pad_to_max_length=True, return_tensors="tf"
            )
        else:
            pytorch_tensor = tokenizer.batch_encode_plus(sequences, pad_to_max_length=True, return_tensors="pt")
            tensorflow_tensor = tokenizer.batch_encode_plus(sequences, pad_to_max_length=True, return_tensors="tf")
            encoded_sequences = tokenizer.batch_encode_plus(sequences, pad_to_max_length=True)

            for key in encoded_sequences.keys():
                pytorch_value = pytorch_tensor[key].tolist()
                tensorflow_value = tensorflow_tensor[key].numpy().tolist()
                encoded_value = encoded_sequences[key]

                self.assertEqual(pytorch_value, tensorflow_value, encoded_value)

    def _check_no_pad_token_padding(self, tokenizer, sequences):
        # if tokenizer does not have pad_token_id, an error should be thrown
        if tokenizer.pad_token_id is None:
            with self.assertRaises(ValueError):
                if isinstance(sequences, list):
                    tokenizer.batch_encode_plus(sequences, pad_to_max_length=True)
                else:
                    tokenizer.encode_plus(sequences, pad_to_max_length=True)

            # add pad_token_id to pass subsequent tests
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    @require_torch
    def test_torch_encode_plus_sent_to_model(self):
        from transformers import MODEL_MAPPING, TOKENIZER_MAPPING

        MODEL_TOKENIZER_MAPPING = merge_model_tokenizer_mappings(MODEL_MAPPING, TOKENIZER_MAPPING)

        tokenizer = self.get_tokenizer()

        if tokenizer.__class__ not in MODEL_TOKENIZER_MAPPING:
            return

        config_class, model_class = MODEL_TOKENIZER_MAPPING[tokenizer.__class__]
        config = config_class()

        if config.is_encoder_decoder or config.pad_token_id is None:
            return

        model = model_class(config)

        # Make sure the model contains at least the full vocabulary size in its embedding matrix
        is_using_common_embeddings = hasattr(model.get_input_embeddings(), "weight")
        assert (model.get_input_embeddings().weight.shape[0] >= len(tokenizer)) if is_using_common_embeddings else True

        # Build sequence
        first_ten_tokens = list(tokenizer.get_vocab().keys())[:10]
        sequence = " ".join(first_ten_tokens)
        encoded_sequence = tokenizer.encode_plus(sequence, return_tensors="pt")
        batch_encoded_sequence = tokenizer.batch_encode_plus([sequence, sequence], return_tensors="pt")
        # This should not fail
        model(**encoded_sequence)
        model(**batch_encoded_sequence)

        if self.test_rust_tokenizer:
            fast_tokenizer = self.get_rust_tokenizer()
            encoded_sequence_fast = fast_tokenizer.encode_plus(sequence, return_tensors="pt")
            batch_encoded_sequence_fast = fast_tokenizer.batch_encode_plus([sequence, sequence], return_tensors="pt")
            # This should not fail
            model(**encoded_sequence_fast)
            model(**batch_encoded_sequence_fast)

    @require_tf
    def test_tf_encode_plus_sent_to_model(self):
        from transformers import TF_MODEL_MAPPING, TOKENIZER_MAPPING

        MODEL_TOKENIZER_MAPPING = merge_model_tokenizer_mappings(TF_MODEL_MAPPING, TOKENIZER_MAPPING)

        tokenizer = self.get_tokenizer()

        if tokenizer.__class__ not in MODEL_TOKENIZER_MAPPING:
            return

        config_class, model_class = MODEL_TOKENIZER_MAPPING[tokenizer.__class__]
        config = config_class()

        if config.is_encoder_decoder or config.pad_token_id is None:
            return

        model = model_class(config)

        # Make sure the model contains at least the full vocabulary size in its embedding matrix
        assert model.config.vocab_size >= len(tokenizer)

        # Build sequence
        first_ten_tokens = list(tokenizer.get_vocab().keys())[:10]
        sequence = " ".join(first_ten_tokens)
        encoded_sequence = tokenizer.encode_plus(sequence, return_tensors="tf")
        batch_encoded_sequence = tokenizer.batch_encode_plus([sequence, sequence], return_tensors="tf")

        # This should not fail
        model(encoded_sequence)
        model(batch_encoded_sequence)

        if self.test_rust_tokenizer:
            fast_tokenizer = self.get_rust_tokenizer()
            encoded_sequence_fast = fast_tokenizer.encode_plus(sequence, return_tensors="tf")
            batch_encoded_sequence_fast = fast_tokenizer.batch_encode_plus([sequence, sequence], return_tensors="tf")
            # This should not fail
            model(encoded_sequence_fast)
            model(batch_encoded_sequence_fast)
