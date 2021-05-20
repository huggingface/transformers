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


import unittest

from transformers import is_flax_available, require_flax
from transformers.testing_utils import require_torch, torch_device
import numpy as np

from .test_modeling_flax_common import ids_tensor


if is_torch_available():
    import jax
    import jax.numpy as jnp

    from transformers.generation_flax_logits_process import (
        FlaxTemperatureLogitsWarper,
        FlaxTopKLogitsWarper,
        FlaxTopPLogitsWarper,
    )


@require_flax
class LogitsProcessorTest(unittest.TestCase):
    def _get_uniform_logits(self, batch_size: int, length: int):
        scores = np.ones((batch_size, length)) / length
        return scores

    def test_temperature_dist_warper(self):
        input_ids = None
        length = 20

        scores = self._get_uniform_logits(batch_size=2, length=length)

        # tweak scores to not be uniform anymore
        scores[1, 5] = (1 / length) + 0.1  # peak, 1st batch
        scores[1, 10] = (1 / length) - 0.4  # valley, 1st batch

        # compute softmax
        probs = jax.nn.softmax(scores, dim=-1)

        temp_dist_warper_sharper = FlaxTemperatureLogitsWarper(temperature=0.5)
        temp_dist_warper_smoother = FlaxTemperatureLogitsWarper(temperature=1.3)

        warped_prob_sharp = jax.nn.softmax(temp_dist_warper_sharper(input_ids, scores.copy()), dim=-1)
        warped_prob_smooth = jax.nn.softmax(temp_dist_warper_smoother(input_ids, scores.copy()), dim=-1)

        # uniform distribution stays uniform
        self.assertTrue(jnp.allclose(probs[0, :], warped_prob_sharp[0, :], atol=1e-3))
        self.assertTrue(jnp.allclose(probs[0, :], warped_prob_smooth[0, :], atol=1e-3))

        # sharp peaks get higher, valleys get lower
        self.assertLess(probs[1, :].max(), warped_prob_sharp[1, :].max())
        self.assertGreater(probs[1, :].min(), warped_prob_sharp[1, :].min())

        # smooth peaks get lower, valleys get higher
        self.assertGreater(probs[1, :].max(), warped_prob_smooth[1, :].max())
        self.assertLess(probs[1, :].min(), warped_prob_smooth[1, :].min())

    def test_top_k_dist_warper(self):
        itorchnput_ids = None
        vocab_size = 10
        batch_size = 2

        # create ramp distribution
        ramp_logits = (
            torch.arange(vocab_size, device=torch_device, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
        )
        ramp_logits[1:, : vocab_size // 2] = ramp_logits[1:, : vocab_size // 2] + vocab_size

        top_k_warp = TopKLogitsWarper(3)

        scores = top_k_warp(input_ids, ramp_logits)

        # check that correct tokens are filtered
        self.assertListEqual(torch.isinf(scores[0]).tolist(), 7 * [True] + 3 * [False])
        self.assertListEqual(torch.isinf(scores[1]).tolist(), 2 * [True] + 3 * [False] + 5 * [True])

        # check special cases
        length = 5

        logits = self._get_uniform_logits(batch_size=batch_size, length=length)
        top_k_warp_safety_check = TopKLogitsWarper(top_k=1, filter_value=0.0, min_tokens_to_keep=3)

        scores = top_k_warp_safety_check(input_ids, logits)
        # uniform dist is not changed
        self.assertListEqual((scores == 0.0).to(torch.long).sum(dim=-1).tolist(), [0, 0])

        ramp_logits = torch.arange(length, device=torch_device, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
        scores = top_k_warp_safety_check(input_ids, ramp_logits)

        # min_tokens overwrites k: 3 tokens are kept => 2 tokens are nullified
        self.assertListEqual((scores == 0.0).to(torch.long).sum(dim=-1).tolist(), [2, 2])

    def test_top_p_dist_warper(self):
        input_ids = None
        vocab_size = 10
        batch_size = 2

        # create distribution and take log (inverse to Softmax as taken in TopPLogitsWarper)
        dist = torch.log(
            torch.tensor([[0.3, 0.1, 0.1, 0.5], [0.15, 0.3, 0.3, 0.25]], device=torch_device, dtype=torch.float)
        )

        top_p_warp = TopPLogitsWarper(0.7)
        filtered_dist = torch.exp(top_p_warp(input_ids, dist))

        # dist should be filtered to keep min num values so that sum is >= 0.7
        # exp (-inf) => 0
        EXPECTED_FILTERED_DIST = torch.tensor(
            [[0.3, 0.0, 0.0, 0.5], [0.0, 0.3, 0.3, 0.25]], device=torch_device, dtype=torch.float
        )
        self.assertTrue(torch.allclose(filtered_dist, EXPECTED_FILTERED_DIST, atol=1e-3))

        # check edge cases with negative and extreme logits
        ramp_logits = torch.arange(vocab_size, device=torch_device, dtype=torch.float).unsqueeze(0).repeat(
            batch_size, 1
        ) - (vocab_size // 2)

        # make ramp_logits more extreme
        ramp_logits[1] = ramp_logits[1] * 100.0

        # make sure at least 2 tokens are kept
        top_p_warp = TopPLogitsWarper(0.9, min_tokens_to_keep=2, filter_value=0.0)
        filtered_dist = top_p_warp(input_ids, ramp_logits)

        # first batch should keep three tokens, second batch would keep only 1, but due to `min_tokens_to_keep=2` keeps 2.
        self.assertListEqual((filtered_dist != 0.0).to(torch.long).sum(dim=-1).tolist(), [3, 2])

    def test_processor_list(self):
        batch_size = 4
        sequence_length = 10
        vocab_size = 15
        eos_token_id = 0

        # dummy input_ids and scores
        input_ids = ids_tensor((batch_size, sequence_length), vocab_size)
        input_ids_comp = input_ids.clone()

        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_comp = scores.clone()

        # instantiate all dist processors
        min_dist_proc = MinLengthLogitsProcessor(min_length=10, eos_token_id=eos_token_id)
        temp_dist_warp = TemperatureLogitsWarper(temperature=0.5)
        rep_penalty_proc = RepetitionPenaltyLogitsProcessor(penalty=2.0)
        top_k_warp = TopKLogitsWarper(3)
        top_p_warp = TopPLogitsWarper(0.8)
        no_repeat_proc = NoRepeatNGramLogitsProcessor(2)
        no_bad_words_dist_proc = NoBadWordsLogitsProcessor(bad_words_ids=[[1]], eos_token_id=eos_token_id)

        # no processor list
        scores = min_dist_proc(input_ids, scores)
        scores = temp_dist_warp(input_ids, scores)
        scores = rep_penalty_proc(input_ids, scores)
        scores = top_k_warp(input_ids, scores)
        scores = top_p_warp(input_ids, scores)
        scores = no_repeat_proc(input_ids, scores)
        scores = no_bad_words_dist_proc(input_ids, scores)

        # with processor list
        processor = LogitsProcessorList(
            [
                min_dist_proc,
                temp_dist_warp,
                rep_penalty_proc,
                top_k_warp,
                top_p_warp,
                no_repeat_proc,
                no_bad_words_dist_proc,
            ]
        )
        scores_comp = processor(input_ids, scores_comp)

        # scores should be equal
        self.assertTrue(torch.allclose(scores, scores_comp, atol=1e-3))

        # input_ids should never be changed
        self.assertListEqual(input_ids.tolist(), input_ids_comp.tolist())
