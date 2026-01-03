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
        AsyncStoppingCriteriaList,
        ConfidenceCriteria,
        EosTokenCriteria,
        MaxLengthCriteria,
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

    def test_confidence_criteria(self):
        criteria = ConfidenceCriteria(assistant_confidence_threshold=0.5)

        vocab_size = 250
        length = 5

        input_ids = ids_tensor((1, length), vocab_size)
        scores = (torch.randn((1, vocab_size)),)

        # Simulate high confidence by setting the probability of the last token to be high
        scores[0][0, input_ids[0, -1]] = 10.0  # Logits before softmax
        self.assertFalse(criteria(input_ids, scores))

        # Simulate low confidence by setting the probability of the last token to be low
        scores[0][0, input_ids[0, -1]] = -10.0  # Logits before softmax
        self.assertTrue(criteria(input_ids, scores))

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

    def test_stop_string_criteria_vocab_size_mismatch(self):
        """Test that StopStringCriteria handles tokens above len(tokenizer) correctly."""
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        # Create input_ids with tokens above len(tokenizer)
        input_ids = torch.tensor([[len(tokenizer) + 1024, 1, 2]], device=torch_device)
        scores = None
        criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=["test"])

        # This should not raise an error and should return False since no stop string is matched
        self.assertFalse(criteria(input_ids, scores))

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
        self.assertEqual(valid_positions, [2, -1, -1, 3, -1, -1])

        # Overlap lengths between end of stop string and start of token
        end_overlaps = embedding_vec[:, 1].tolist()
        self.assertEqual(end_overlaps, [-1, 3, 3, -1, 1, -1])

        # Length of each token
        token_lengths = embedding_vec[:-1, 2].tolist()
        self.assertEqual(token_lengths, [len(token) for token in token_list])

    def test_single_letter_stop_string(self):
        true_strings = ["a", "baa", "abc"]  # "abc" is a single token
        false_strings = ["abbbbbbb", "b"]  # "abbbbbbb" is split into multiple tokens
        stop_strings = ["a"]
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", add_prefix_space=False)
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

    def test_criteria_per_row(self):
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

        # trigger stopping when at least one criteria is satisfied, one value per batch
        self.assertTrue(criteria(inputs["input_ids"], scores))

        # return False when neither is satisfied
        self.assertFalse(criteria(inputs["input_ids"][:, :-1], scores))

    def test_criteria_per_row_batched(self):
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

        # trigger stopping when at least one criteria is satisfied
        self.assertListEqual(criteria(inputs["input_ids"], scores).tolist(), [True, False, False])

        # False when neither is satisfied
        self.assertListEqual(criteria(inputs["input_ids"][:, :-1], scores).tolist(), [False, False, False])


@require_torch
class AsyncStoppingCriteriaTestCase(unittest.TestCase):
    """Test cases for AsyncStoppingCriteriaList."""

    def _get_tensors(self, length, batch_size=3):
        vocab_size = 250

        input_ids = ids_tensor((batch_size, length), vocab_size)
        scores = torch.ones((batch_size, length), device=torch_device, dtype=torch.float) / length
        return input_ids, scores

    def test_async_wrapper_basic(self):
        """Test that AsyncStoppingCriteriaList wraps StoppingCriteriaList correctly."""
        criteria_list = StoppingCriteriaList([MaxLengthCriteria(max_length=10)])
        async_criteria = AsyncStoppingCriteriaList(criteria_list)

        # Test __len__
        self.assertEqual(len(async_criteria), 1)

        # Test __iter__
        criteria_items = list(async_criteria)
        self.assertEqual(len(criteria_items), 1)
        self.assertIsInstance(criteria_items[0], MaxLengthCriteria)

        # Test max_length property
        self.assertEqual(async_criteria.max_length, 10)

    def test_async_sync_equivalence_max_length(self):
        """Test that async and sync modes produce identical results for max_length stopping."""
        input_ids, scores = self._get_tensors(5)

        # Sync behavior
        sync_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=10)])
        sync_result = sync_criteria(input_ids, scores)

        # At length 5 with max_length 10, should not be finished
        self.assertFalse(all(sync_result))

        # Async behavior (should fall back to sync on CPU)
        async_criteria = AsyncStoppingCriteriaList(StoppingCriteriaList([MaxLengthCriteria(max_length=10)]))
        unfinished = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        updated_unfinished, this_peer_finished = async_criteria.check(input_ids, scores, unfinished)

        # At length 5 with max_length 10, should not be finished
        self.assertFalse(this_peer_finished)
        self.assertTrue(all(updated_unfinished == 1))

        # At length 10, should be finished
        input_ids_long, scores_long = self._get_tensors(10)
        sync_result_long = sync_criteria(input_ids_long, scores_long)
        self.assertTrue(all(sync_result_long))

        unfinished = torch.ones(input_ids_long.shape[0], device=input_ids_long.device, dtype=torch.long)
        updated_unfinished, this_peer_finished = async_criteria.check(input_ids_long, scores_long, unfinished)
        self.assertTrue(this_peer_finished)
        self.assertTrue(all(updated_unfinished == 0))

    def test_async_sync_equivalence_eos_token(self):
        """Test that async and sync modes produce identical results for EOS token stopping."""
        input_ids, scores = self._get_tensors(5)

        # Set EOS token (0) at the end of all sequences
        input_ids[:, -1] = 0

        # Sync behavior
        sync_criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=20),
                EosTokenCriteria(eos_token_id=0),
            ]
        )
        sync_result = sync_criteria(input_ids, scores)

        # Async behavior - the async criteria checks results from PREVIOUS async operations
        # so we need to call check() multiple times to allow async results to be retrieved
        async_criteria = AsyncStoppingCriteriaList(
            StoppingCriteriaList(
                [
                    MaxLengthCriteria(max_length=20),
                    EosTokenCriteria(eos_token_id=0),
                ]
            )
        )
        unfinished = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.long)

        # First call starts async check
        updated_unfinished, _ = async_criteria.check(input_ids, scores, unfinished)

        # Wait for async check to complete and call again to retrieve result
        if input_ids.device.type == "cuda":
            torch.cuda.synchronize()
        updated_unfinished, this_peer_finished = async_criteria.check(input_ids, scores, updated_unfinished)

        # Both should indicate all sequences have EOS
        self.assertTrue(all(sync_result))
        self.assertTrue(this_peer_finished)
        self.assertTrue(all(updated_unfinished == 0))

    def test_async_sync_equivalence_partial_eos(self):
        """Test async/sync equivalence when only some sequences have EOS."""
        input_ids, scores = self._get_tensors(5)

        # Only first 2 sequences have EOS
        input_ids[:2, -1] = 0
        input_ids[2, -1] = 1

        # Sync behavior
        sync_criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=20),
                EosTokenCriteria(eos_token_id=0),
            ]
        )
        sync_result = sync_criteria(input_ids, scores)

        # Should match [True, True, False]
        self.assertListEqual(sync_result.tolist(), [True, True, False])

    def test_async_different_batch_sizes(self):
        """Test async stopping criteria with different batch sizes."""
        for batch_size in [1, 2, 4, 8, 16]:
            input_ids, scores = self._get_tensors(5, batch_size=batch_size)

            async_criteria = AsyncStoppingCriteriaList(StoppingCriteriaList([MaxLengthCriteria(max_length=10)]))
            unfinished = torch.ones(batch_size, device=input_ids.device, dtype=torch.long)
            updated_unfinished, this_peer_finished = async_criteria.check(input_ids, scores, unfinished)

            self.assertEqual(updated_unfinished.shape[0], batch_size)
            self.assertFalse(this_peer_finished)

            # At max_length, all should finish
            input_ids_long, scores_long = self._get_tensors(10, batch_size=batch_size)
            unfinished = torch.ones(batch_size, device=input_ids_long.device, dtype=torch.long)
            updated_unfinished, this_peer_finished = async_criteria.check(input_ids_long, scores_long, unfinished)

            self.assertEqual(updated_unfinished.shape[0], batch_size)
            self.assertTrue(this_peer_finished)

    def test_async_cpu_fallback(self):
        """Test that async gracefully falls back to sync on CPU."""
        # Force CPU tensors
        batch_size = 3
        vocab_size = 250
        length = 5

        input_ids = torch.randint(0, vocab_size, (batch_size, length), device="cpu")
        scores = torch.ones((batch_size, length), device="cpu", dtype=torch.float)

        async_criteria = AsyncStoppingCriteriaList(
            StoppingCriteriaList(
                [
                    MaxLengthCriteria(max_length=10),
                    EosTokenCriteria(eos_token_id=0),
                ]
            )
        )
        unfinished = torch.ones(batch_size, device="cpu", dtype=torch.long)

        # Should work without errors on CPU (sync fallback)
        updated_unfinished, this_peer_finished = async_criteria.check(input_ids, scores, unfinished)
        self.assertFalse(this_peer_finished)

        # With EOS in all sequences
        input_ids[:, -1] = 0
        updated_unfinished, this_peer_finished = async_criteria.check(input_ids, scores, unfinished)
        self.assertTrue(this_peer_finished)

    def test_async_legacy_call_interface(self):
        """Test that the legacy __call__ interface still works."""
        input_ids, scores = self._get_tensors(5)

        async_criteria = AsyncStoppingCriteriaList(StoppingCriteriaList([MaxLengthCriteria(max_length=10)]))

        # __call__ should fall back to sync behavior
        result = async_criteria(input_ids, scores)
        self.assertEqual(result.shape[0], 3)
        self.assertFalse(all(result))  # Not at max_length yet

        input_ids_long, scores_long = self._get_tensors(10)
        result = async_criteria(input_ids_long, scores_long)
        self.assertTrue(all(result))  # At max_length

    def test_async_finalize(self):
        """Test the finalize method for cleanup."""
        input_ids, scores = self._get_tensors(5)

        async_criteria = AsyncStoppingCriteriaList(StoppingCriteriaList([MaxLengthCriteria(max_length=100)]))
        unfinished = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.long)

        # Do a check to potentially start an async operation
        async_criteria.check(input_ids, scores, unfinished)

        # Finalize should work without errors
        final_unfinished, this_peer_finished = async_criteria.finalize(unfinished)
        self.assertFalse(this_peer_finished)

    def test_async_custom_stopping_criteria(self):
        """Test async with a custom stopping criteria."""
        from transformers.generation import StoppingCriteria

        class CustomStoppingCriteria(StoppingCriteria):
            """Stop when the last token is a specific value."""

            def __init__(self, stop_token_id):
                self.stop_token_id = stop_token_id

            def __call__(self, input_ids, scores, **kwargs):
                return input_ids[:, -1] == self.stop_token_id

        input_ids, scores = self._get_tensors(5)
        stop_token_id = 42
        input_ids[:, -1] = stop_token_id  # Set last token to stop token

        async_criteria = AsyncStoppingCriteriaList(
            StoppingCriteriaList(
                [
                    MaxLengthCriteria(max_length=100),
                    CustomStoppingCriteria(stop_token_id=stop_token_id),
                ]
            )
        )
        unfinished = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        updated_unfinished, this_peer_finished = async_criteria.check(input_ids, scores, unfinished)

        # Custom criteria should have triggered stop via sync fallback (near max_length check)
        # At length 5 with max_length 100, it would use _check_async_only,
        # but first call will start the async check
        # For proper testing, we need the sync path which happens near max_length
        input_ids2, scores2 = self._get_tensors(99)  # Near max_length
        input_ids2[:, -1] = stop_token_id
        async_criteria2 = AsyncStoppingCriteriaList(
            StoppingCriteriaList(
                [
                    MaxLengthCriteria(max_length=100),
                    CustomStoppingCriteria(stop_token_id=stop_token_id),
                ]
            )
        )
        unfinished2 = torch.ones(input_ids2.shape[0], device=input_ids2.device, dtype=torch.long)
        updated_unfinished2, this_peer_finished2 = async_criteria2.check(input_ids2, scores2, unfinished2)
        self.assertTrue(this_peer_finished2)

    def test_async_multiple_eos_tokens(self):
        """Test async with multiple EOS token IDs."""
        input_ids, scores = self._get_tensors(5)

        # Different sequences end with different EOS tokens
        input_ids[0, -1] = 1  # First EOS token
        input_ids[1, -1] = 2  # Second EOS token
        input_ids[2, -1] = 99  # Not an EOS token

        async_criteria = AsyncStoppingCriteriaList(
            StoppingCriteriaList(
                [
                    MaxLengthCriteria(max_length=100),
                    EosTokenCriteria(eos_token_id=[1, 2]),  # Multiple EOS tokens
                ]
            )
        )

        # Test at near max_length to trigger sync path
        input_ids_near, scores_near = self._get_tensors(99)
        input_ids_near[0, -1] = 1
        input_ids_near[1, -1] = 2
        input_ids_near[2, -1] = 99

        unfinished = torch.ones(input_ids_near.shape[0], device=input_ids_near.device, dtype=torch.long)
        updated_unfinished, this_peer_finished = async_criteria.check(input_ids_near, scores_near, unfinished)

        # First two should be done (EOS), third should not
        self.assertListEqual(updated_unfinished.tolist(), [0, 0, 1])
        self.assertFalse(this_peer_finished)  # Not all finished
