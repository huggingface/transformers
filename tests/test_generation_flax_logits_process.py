# coding=utf-8
# Copyright 2021 The HuggingFace Team Inc.
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

import numpy as np

from transformers import is_flax_available
from transformers.testing_utils import require_flax

from .test_modeling_flax_common import ids_tensor


if is_flax_available():
    import jax
    import jax.numpy as jnp
    from transformers.generation_flax_logits_process import (
        FlaxLogitsProcessorList,
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
        probs = jax.nn.softmax(scores, axis=-1)

        temp_dist_warper_sharper = FlaxTemperatureLogitsWarper(temperature=0.5)
        temp_dist_warper_smoother = FlaxTemperatureLogitsWarper(temperature=1.3)

        warped_prob_sharp = jax.nn.softmax(temp_dist_warper_sharper(input_ids, scores.copy()), axis=-1)
        warped_prob_smooth = jax.nn.softmax(temp_dist_warper_smoother(input_ids, scores.copy()), axis=-1)

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
        input_ids = None
        vocab_size = 10
        batch_size = 2

        # create ramp distribution
        ramp_logits = np.broadcast_to(np.arange(vocab_size)[None, :], (batch_size, vocab_size)).copy()
        ramp_logits[1:, : vocab_size // 2] = ramp_logits[1:, : vocab_size // 2] + vocab_size

        top_k_warp = FlaxTopKLogitsWarper(3)

        scores = top_k_warp(input_ids, ramp_logits)

        # check that correct tokens are filtered
        self.assertListEqual(jnp.isinf(scores[0]).tolist(), 7 * [True] + 3 * [False])
        self.assertListEqual(jnp.isinf(scores[1]).tolist(), 2 * [True] + 3 * [False] + 5 * [True])

        # check special case
        length = 5
        top_k_warp_safety_check = FlaxTopKLogitsWarper(top_k=1, filter_value=0.0, min_tokens_to_keep=3)

        ramp_logits = np.broadcast_to(np.arange(length)[None, :], (batch_size, length)).copy()
        scores = top_k_warp_safety_check(input_ids, ramp_logits)

        # min_tokens overwrites k: 3 tokens are kept => 2 tokens are nullified
        self.assertListEqual((scores == 0.0).sum(axis=-1).tolist(), [2, 2])

    def test_top_p_dist_warper(self):
        input_ids = None
        vocab_size = 10
        batch_size = 2

        # create distribution and take log (inverse to Softmax as taken in TopPLogitsWarper)
        dist = np.log(np.array([[0.3, 0.1, 0.1, 0.5], [0.15, 0.3, 0.3, 0.25]]))

        top_p_warp = FlaxTopPLogitsWarper(0.7)
        filtered_dist = np.exp(top_p_warp(input_ids, dist))

        # dist should be filtered to keep min num values so that sum is >= 0.7
        # exp (-inf) => 0
        EXPECTED_FILTERED_DIST = np.array([[0.3, 0.0, 0.0, 0.5], [0.0, 0.3, 0.3, 0.25]])
        self.assertTrue(np.allclose(filtered_dist, EXPECTED_FILTERED_DIST, atol=1e-3))

        # check edge cases with negative and extreme logits
        ramp_logits = np.broadcast_to(np.arange(vocab_size)[None, :], (batch_size, vocab_size)).copy() - (
            vocab_size // 2
        )

        # make ramp_logits more extreme
        ramp_logits[1] = ramp_logits[1] * 100.0

        # make sure at least 2 tokens are kept
        top_p_warp = FlaxTopPLogitsWarper(0.9, min_tokens_to_keep=2, filter_value=0.0)
        filtered_dist = top_p_warp(input_ids, ramp_logits)

        # first batch should keep three tokens, second batch would keep only 1, but due to `min_tokens_to_keep=2` keeps 2.
        self.assertListEqual((filtered_dist != 0.0).sum(axis=-1).tolist(), [3, 2])

    def test_processor_list(self):
        batch_size = 4
        sequence_length = 10
        vocab_size = 15

        # dummy input_ids and scores
        input_ids = ids_tensor((batch_size, sequence_length), vocab_size)
        input_ids_comp = input_ids.copy()

        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_comp = scores.copy()

        # instantiate all dist processors
        temp_dist_warp = FlaxTemperatureLogitsWarper(temperature=0.5)
        top_k_warp = FlaxTopKLogitsWarper(3)
        top_p_warp = FlaxTopPLogitsWarper(0.8)

        # no processor list
        scores = temp_dist_warp(input_ids, scores)
        scores = top_k_warp(input_ids, scores)
        scores = top_p_warp(input_ids, scores)

        # with processor list
        processor = FlaxLogitsProcessorList([temp_dist_warp, top_k_warp, top_p_warp])
        scores_comp = processor(input_ids, scores_comp)

        # scores should be equal
        self.assertTrue(jnp.allclose(scores, scores_comp, atol=1e-3))

        # input_ids should never be changed
        self.assertListEqual(input_ids.tolist(), input_ids_comp.tolist())
