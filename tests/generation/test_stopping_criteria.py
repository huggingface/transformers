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

from transformers.generation.stopping_criteria import (
    _stop_string_create_embedding_vecs,
    _stop_string_get_matching_positions,
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
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        stop_strings = ["aaaaaaa", "assdfiugsdf", "stop"]
        criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings)
        idx_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
        all_token_valid_positions, all_token_end_overlaps = _stop_string_get_matching_positions(
            tok_list=criteria.tok_list, tok_indices=criteria.tok_indices, stop_strings=criteria.stop_strings
        )
        for stop_string in stop_strings:
            token_valid_positions = all_token_valid_positions[stop_string]
            token_end_overlaps = all_token_end_overlaps[stop_string]
            for token_idx, valid_positions in token_valid_positions.items():
                token = idx_to_token[token_idx].replace("▁", " ").replace("Ġ", " ")
                for position in valid_positions:
                    trim_length = position + len(token) - len(stop_string)
                    if trim_length > 0:
                        # This token runs off the start of the string
                        self.assertTrue(stop_string.startswith(token[trim_length:]))
                    else:
                        self.assertTrue(stop_string[-position - len(token) :].startswith(token))
            for token_idx, end_overlaps in token_end_overlaps.items():
                token = idx_to_token[token_idx].replace("▁", " ").replace("Ġ", " ")
                for overlap in end_overlaps:
                    # Either this token runs off the end of the string,
                    # or the entire stop string is a substring of the token
                    self.assertTrue(
                        (
                            stop_string.endswith(token[:overlap])
                            or (stop_string in token and overlap == len(stop_string))
                        )
                    )

    def test_stop_string_embedding_vecs(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        stop_strings = ["aaaaaaa", "assdfiugsdf", "stop"]
        criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings)
        all_embedding_vecs, *_ = _stop_string_create_embedding_vecs(
            tok_list=criteria.tok_list, tok_indices=criteria.tok_indices, stop_strings=criteria.stop_strings
        )
        for stop_string in stop_strings:
            embedding_vecs = all_embedding_vecs[stop_string]
            max_valid_positions = criteria.max_valid_positions[stop_string]
            max_valid_end_lens = criteria.max_valid_end_lens[stop_string]
            for token, token_idx in zip(criteria.tok_list, criteria.tok_indices):
                vec = embedding_vecs[token_idx].tolist()
                # The embedding contains packed valid positions, end overlap lengths, and the total token length
                token = token.replace("▁", " ").replace("Ġ", " ")

                token_valid_positions = vec[:max_valid_positions]
                for position in token_valid_positions:
                    if position == -1:
                        continue  # Padding value
                    trim_length = position + len(token) - len(stop_string)
                    if trim_length > 0:
                        # This token runs off the start of the string
                        self.assertTrue(stop_string.startswith(token[trim_length:]))
                    else:
                        self.assertTrue(stop_string[-position - len(token) : -position] == token)

                token_end_overlaps = vec[max_valid_positions : max_valid_positions + max_valid_end_lens]
                for overlap in token_end_overlaps:
                    if overlap == -1:
                        continue  # Padding value
                    # Either this token runs off the end of the string,
                    # or the entire stop string is a substring of the token
                    self.assertTrue(
                        (
                            stop_string.endswith(token[:overlap])
                            or (stop_string in token and overlap == len(stop_string))
                        )
                    )

                token_length = vec[-1]
                self.assertTrue(len(token) == token_length)
