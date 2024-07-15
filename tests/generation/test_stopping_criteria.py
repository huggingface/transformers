# coding=utf-8
# Copyright 2020 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import unittest

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ..test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers.generation import (
        EosTokenCriteria,
        MaxLengthCriteria,
        MaxNewTokensCriteria,
        MaxTimeCriteria,
        StoppingCriteriaList,
        StopStringCriteria,
        validate_stopping_criteria,
    )


@require_torch
class StoppingCriteriaTestCase(unittest.TestCase):
    def _get_tensors(self, length):
        batch_size = 3
        vocab_size = 250

        input_ids = ids_tensor((batch_size, length), vocab_size)
        scores = torch.ones((batch_size, length), device=torch_device, dtype=torch.float) / length
        return input_ids, scores

    def test_list_criteria(self):
        input_ids, scores = self._get_tensors(5)

        criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=10),
                MaxTimeCriteria(max_time=0.1),
            ]
        )

        self.assertFalse(all(criteria(input_ids, scores)))

        input_ids, scores = self._get_tensors(9)
        self.assertFalse(all(criteria(input_ids, scores)))

        input_ids, scores = self._get_tensors(10)
        self.assertTrue(all(criteria(input_ids, scores)))

    def test_max_length_criteria(self):
        criteria = MaxLengthCriteria(max_length=10)

        input_ids, scores = self._get_tensors(5)
        self.assertFalse(all(criteria(input_ids, scores)))

        input_ids, scores = self._get_tensors(9)
        self.assertFalse(all(criteria(input_ids, scores)))

        input_ids, scores = self._get_tensors(10)
        self.assertTrue(all(criteria(input_ids, scores)))

    def test_max_new_tokens_criteria(self):
        criteria = MaxNewTokensCriteria(start_length=5, max_new_tokens=5)

        input_ids, scores = self._get_tensors(5)
        self.assertFalse(all(criteria(input_ids, scores)))

        input_ids, scores = self._get_tensors(9)
        self.assertFalse(all(criteria(input_ids, scores)))

        input_ids, scores = self._get_tensors(10)
        self.assertTrue(all(criteria(input_ids, scores)))

        criteria_list = StoppingCriteriaList([criteria])
        self.assertEqual(criteria_list.max_length, 10)

    def test_max_time_criteria(self):
        input_ids, scores = self._get_tensors(5)

        criteria = MaxTimeCriteria(max_time=0.1)
        self.assertFalse(all(criteria(input_ids, scores)))

        criteria = MaxTimeCriteria(max_time=0.1, initial_timestamp=time.time() - 0.2)
        self.assertTrue(all(criteria(input_ids, scores)))

    def test_eos_token_criteria(self):
        criteria = EosTokenCriteria(eos_token_id=0)

        input_ids, scores = self._get_tensors(5)
        input_ids[:, -1] = 0
        self.assertTrue(all(criteria(input_ids, scores)))

        input_ids, scores = self._get_tensors(5)
        input_ids[:2, -1] = 0
        input_ids[2, -1] = 1
        self.assertListEqual(criteria(input_ids, scores).tolist(), [True, True, False])

        input_ids, scores = self._get_tensors(5)
        input_ids[:, -1] = 1
        self.assertListEqual(criteria(input_ids, scores).tolist(), [False, False, False])

    def test_validate_stopping_criteria(self):
        validate_stopping_criteria(StoppingCriteriaList([MaxLengthCriteria(10)]), 10)

        with self.assertWarns(UserWarning):
            validate_stopping_criteria(StoppingCriteriaList([MaxLengthCriteria(10)]), 11)

        stopping_criteria = validate_stopping_criteria(StoppingCriteriaList(), 11)

        self.assertEqual(len(stopping_criteria), 1)

    def test_stop_string_criteria(self):
        true_strings = [
            "<|im_start|><|im_end|>",
            "<|im_start|><|im_end|<|im_end|>",
            ">><|im_start|>>stop",
            "stop",
            "e nd",
        ]
        false_strings = [
            "<|im_start|><|im_end|",
            "<|im_start|><|im_end|<|im_end|",
            "<|im_end|><|im_start|>",
            "<|im_end|<>stop<|im_end|",
            "end",
            "en d",
            "eNd",
            "<|im_end|",
            "|im_end|>",
            "s",
        ]
        stop_strings = ["<|im_end|>", "stop", "e nd"]

        # Use a tokenizer that won't actually have special tokens for these
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        true_input_ids = tokenizer(true_strings, return_tensors="pt", padding="longest", add_special_tokens=False)
        false_input_ids = tokenizer(false_strings, return_tensors="pt", padding="longest", add_special_tokens=False)

        scores = None
        criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings)
        for i in range(len(true_strings)):
            self.assertTrue(criteria(true_input_ids["input_ids"][i : i + 1], scores))
        for i in range(len(false_strings)):
            self.assertFalse(criteria(false_input_ids["input_ids"][i : i + 1], scores))

        # Now try it with a tokenizer where those are actually special tokens
        tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/dolphin-2.5-mixtral-8x7b")
        tokenizer.padding_side = "left"
        true_input_ids = tokenizer(true_strings, return_tensors="pt", padding="longest", add_special_tokens=False)
        false_input_ids = tokenizer(false_strings, return_tensors="pt", padding="longest", add_special_tokens=False)

        criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings)
        for i in range(len(true_strings)):
            self.assertTrue(criteria(true_input_ids["input_ids"][i : i + 1], scores))
        for i in range(len(false_strings)):
            self.assertFalse(criteria(false_input_ids["input_ids"][i : i + 1], scores))

    def test_stop_string_matching_positions(self):
        stop_string = "stop"
        token_list = ["last", "top", "topper", "s", "p"]
        token_indices = list(range(len(token_list)))
        all_token_valid_positions, all_token_end_overlaps = StopStringCriteria._stop_string_get_matching_positions(
            token_list=token_list, token_indices=token_indices, stop_strings=[stop_string]
        )
        valid_positions = {
            token_list[idx]: positions for idx, positions in all_token_valid_positions[stop_string].items()
        }
        end_overlaps = {token_list[idx]: overlaps for idx, overlaps in all_token_end_overlaps[stop_string].items()}
        self.assertEqual(valid_positions, {"s": [3], "last": [2]})
        self.assertEqual(end_overlaps, {"top": [3], "topper": [3], "p": [1]})

    def test_stop_string_embedding_vecs(self):
        stop_string = "stop"
        token_list = ["last", "top", "topper", "s", "p"]
        token_indices = list(range(len(token_list)))
        embedding_vec, max_valid_positions, max_valid_end_lens = StopStringCriteria._stop_string_create_embedding_vec(
            token_list=token_list, token_indices=token_indices, stop_strings=[stop_string]
        )

        # Positions inside the stop string where the token matches (excluding end overlaps)
        valid_positions = embedding_vec[:, 0].tolist()
        self.assertEqual(valid_positions, [2, -1, -1, 3, -1])

        # Overlap lengths between end of stop string and start of token
        end_overlaps = embedding_vec[:, 1].tolist()
        self.assertEqual(end_overlaps, [-1, 3, 3, -1, 1])

        # Length of each token
        token_lengths = embedding_vec[:, 2].tolist()
        self.assertEqual(token_lengths, [len(token) for token in token_list])

    def test_single_letter_stop_string(self):
        true_strings = ["a", "baa", "abc"]  # "abc" is a single token
        false_strings = ["abbbbbbb", "b"]  # "abbbbbbb" is split into multiple tokens
        stop_strings = ["a"]
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        true_input_ids = tokenizer(true_strings, return_tensors="pt", padding="longest", add_special_tokens=False)
        false_input_ids = tokenizer(false_strings, return_tensors="pt", padding="longest", add_special_tokens=False)

        scores = None
        criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings)
        for input_ids in true_input_ids["input_ids"]:
            self.assertTrue(criteria(input_ids.unsqueeze(0), scores))
        for input_ids in false_input_ids["input_ids"]:
            self.assertFalse(criteria(input_ids.unsqueeze(0), scores))

    def test_criterias_per_row(self):
        text = "They completed the challenging puzzle, revealing the hidden image at the end"
        stop_strings = ["end"]

        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

        scores = None
        criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=20),
                StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings),
            ]
        )

        # trigger stopping when at leat one criteria is satisfied, one value per batch
        self.assertTrue(criteria(inputs["input_ids"], scores))

        # return False when neither is satisfied
        self.assertFalse(criteria(inputs["input_ids"][:, :-1], scores))

    def test_criterias_per_row_batched(self):
        text = [
            "They completed the challenging puzzle, revealing the hidden image at the end",
            "Today a dragon flew over France",
            "The aroma of freshly baked pizza filled the kitchen",
        ]
        stop_strings = ["end"]

        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        inputs = tokenizer(text, return_tensors="pt", padding="longest", add_special_tokens=False)

        scores = None
        criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=20),
                StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings),
            ]
        )

        # trigger stopping when at leat one criteria is satisfied
        self.assertListEqual(criteria(inputs["input_ids"], scores).tolist(), [True, False, False])

        # False when neither is satisfied
        self.assertListEqual(criteria(inputs["input_ids"][:, :-1], scores).tolist(), [False, False, False])
