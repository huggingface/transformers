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


from __future__ import annotations

import unittest

import numpy as np
from parameterized import parameterized

from transformers import is_tf_available
from transformers.testing_utils import require_tf


if is_tf_available():
    import tensorflow as tf

    from transformers.generation import (
        TFForcedBOSTokenLogitsProcessor,
        TFForcedEOSTokenLogitsProcessor,
        TFForceTokensLogitsProcessor,
        TFLogitsProcessorList,
        TFMinLengthLogitsProcessor,
        TFNoBadWordsLogitsProcessor,
        TFNoRepeatNGramLogitsProcessor,
        TFRepetitionPenaltyLogitsProcessor,
        TFSuppressTokensAtBeginLogitsProcessor,
        TFSuppressTokensLogitsProcessor,
        TFTemperatureLogitsWarper,
        TFTopKLogitsWarper,
        TFTopPLogitsWarper,
    )

    from ..test_modeling_tf_common import ids_tensor


@require_tf
class TFLogitsProcessorTest(unittest.TestCase):
    def _get_uniform_logits(self, batch_size: int, length: int):
        scores = tf.ones((batch_size, length), dtype=tf.float32) / length
        return scores

    @parameterized.expand([(False,), (True,)])
    def test_min_length_dist_processor(self, use_xla):
        vocab_size = 20
        batch_size = 4
        eos_token_id = 0

        min_dist_processor = TFMinLengthLogitsProcessor(min_length=10, eos_token_id=eos_token_id)
        if use_xla:
            min_dist_processor = tf.function(min_dist_processor, jit_compile=True)

        # check that min length is applied at length 5
        cur_len = 5
        input_ids = ids_tensor((batch_size, cur_len), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = min_dist_processor(input_ids, scores, cur_len)
        self.assertListEqual(scores_before_min_length[:, eos_token_id].numpy().tolist(), 4 * [-float("inf")])

        # check that min length is not applied anymore at length 15
        cur_len = 15
        input_ids = ids_tensor((batch_size, cur_len), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = min_dist_processor(input_ids, scores, cur_len)
        self.assertFalse(tf.math.reduce_any(tf.math.is_inf(scores_before_min_length)).numpy())

    @parameterized.expand([(False,), (True,)])
    def test_temperature_dist_warper(self, use_xla):
        input_ids = None
        cur_len = None
        length = 20

        scores = self._get_uniform_logits(batch_size=2, length=length)

        # tweak scores to not be uniform anymore
        scores = scores.numpy()
        scores[1, 5] = (1 / length) + 0.1  # peak, 1st batch
        scores[1, 10] = (1 / length) - 0.4  # valley, 1st batch
        scores = tf.convert_to_tensor(scores)

        # compute softmax
        probs = tf.nn.softmax(scores, axis=-1)

        temp_dist_warper_sharper = TFTemperatureLogitsWarper(temperature=0.5)
        temp_dist_warper_smoother = TFTemperatureLogitsWarper(temperature=1.3)
        if use_xla:
            temp_dist_warper_sharper = tf.function(temp_dist_warper_sharper, jit_compile=True)
            temp_dist_warper_smoother = tf.function(temp_dist_warper_smoother, jit_compile=True)

        warped_prob_sharp = tf.nn.softmax(temp_dist_warper_sharper(input_ids, tf.identity(scores), cur_len), axis=-1)
        warped_prob_smooth = tf.nn.softmax(temp_dist_warper_smoother(input_ids, tf.identity(scores), cur_len), axis=-1)

        # uniform distribution stays uniform
        tf.debugging.assert_near(probs[0, :], warped_prob_sharp[0, :], atol=1e-3)
        tf.debugging.assert_near(probs[0, :], warped_prob_smooth[0, :], atol=1e-3)

        # sharp peaks get higher, valleys get lower
        self.assertLess(tf.math.reduce_max(probs[1, :]), tf.math.reduce_max(warped_prob_sharp[1, :]))
        self.assertGreater(tf.math.reduce_min(probs[1, :]), tf.math.reduce_min(warped_prob_sharp[1, :]))

        # smooth peaks get lower, valleys get higher
        self.assertGreater(tf.math.reduce_max(probs[1, :]), tf.math.reduce_max(warped_prob_smooth[1, :]))
        self.assertLess(tf.math.reduce_min(probs[1, :]), tf.math.reduce_min(warped_prob_smooth[1, :]))

    @parameterized.expand([(False,), (True,)])
    def test_repetition_penalty_dist_process(self, use_xla):
        vocab_size = 10
        cur_len = 2

        input_ids = tf.constant([[0, 1], [5, 0]], dtype=tf.int32)
        self.assertEqual(cur_len, input_ids.shape[1])

        scores = self._get_uniform_logits(batch_size=2, length=vocab_size)

        mask = tf.cast(tf.constant([[1] + 9 * [0], 10 * [0]]), tf.bool)
        scores = tf.where(mask, -1 / vocab_size, scores)
        mask = tf.cast(tf.constant([10 * [0], 5 * [0] + [1] + 4 * [0]]), tf.bool)
        scores = tf.where(mask, 4 / vocab_size, scores)
        rep_penalty_proc = TFRepetitionPenaltyLogitsProcessor(penalty=2.0)
        if use_xla:
            rep_penalty_proc = tf.function(rep_penalty_proc, jit_compile=True)

        scores = rep_penalty_proc(input_ids, tf.identity(scores), cur_len)

        # check that values were correctly changed (negative scores for used tokens should increase, others
        # should decrease)
        self.assertAlmostEqual(scores[0, 0].numpy(), -(1 / vocab_size) * 2)
        self.assertAlmostEqual(scores[0, 1].numpy(), (1 / vocab_size) / 2)
        self.assertAlmostEqual(scores[0, 2].numpy(), (1 / vocab_size))  # unused tokens should see no change

        self.assertAlmostEqual(scores[1, 0].numpy(), (1 / vocab_size) / 2)
        self.assertAlmostEqual(scores[1, 5].numpy(), (4 / vocab_size) / 2)
        self.assertAlmostEqual(scores[0, 2].numpy(), (1 / vocab_size))  # unused tokens should see no change

    @parameterized.expand([(False,), (True,)])
    def test_top_k_dist_warper(self, use_xla):
        input_ids = None
        cur_len = None
        vocab_size = 10
        batch_size = 2

        # create ramp distribution
        ramp_logits = np.broadcast_to(np.arange(vocab_size, dtype=np.float32), (batch_size, vocab_size)).copy()
        ramp_logits[1:, : vocab_size // 2] = ramp_logits[1:, : vocab_size // 2] + vocab_size

        top_k_warp = TFTopKLogitsWarper(3)
        if use_xla:
            top_k_warp = tf.function(top_k_warp, jit_compile=True)

        scores = top_k_warp(input_ids, ramp_logits, cur_len)

        # check that correct tokens are filtered
        self.assertListEqual(tf.math.is_inf(scores[0]).numpy().tolist(), 7 * [True] + 3 * [False])
        self.assertListEqual(tf.math.is_inf(scores[1]).numpy().tolist(), 2 * [True] + 3 * [False] + 5 * [True])

        # check special cases
        length = 5

        logits = self._get_uniform_logits(batch_size=batch_size, length=length)
        top_k_warp_safety_check = TFTopKLogitsWarper(top_k=1, filter_value=0.0, min_tokens_to_keep=3)
        if use_xla:
            top_k_warp_safety_check = tf.function(top_k_warp_safety_check, jit_compile=True)

        scores = top_k_warp_safety_check(input_ids, logits, cur_len)
        # uniform dist is not changed
        self.assertListEqual(tf.math.reduce_sum(tf.where(scores == 0.0, 1, 0), axis=-1).numpy().tolist(), [0, 0])

        ramp_logits = np.broadcast_to(np.arange(length, dtype=np.float32), (batch_size, length)).copy()
        scores = top_k_warp_safety_check(input_ids, ramp_logits, cur_len)

        # min_tokens overwrites k: 3 tokens are kept => 2 tokens are nullified
        self.assertListEqual(tf.math.reduce_sum(tf.where(scores == 0.0, 1, 0), axis=-1).numpy().tolist(), [2, 2])

    @parameterized.expand([(False,), (True,)])
    def test_top_p_dist_warper(self, use_xla):
        input_ids = None
        cur_len = None
        vocab_size = 10
        batch_size = 2

        # create distribution and take log (inverse to Softmax as taken in TFTopPLogitsWarper)
        dist = np.log(np.array([[0.3, 0.1, 0.1, 0.5], [0.15, 0.3, 0.3, 0.25]], dtype=np.float32))

        # top_p should have been 0.8 to test the edge case of top_p being exactly equal to sum of some token prob
        # However, due to the numerical instability of softmax in TF we choose this as the edge case
        # top_p as 0.8 passes when use_xla is True and fails when False. Refer PR #18984.
        top_p_warp = TFTopPLogitsWarper(0.79999995)
        if use_xla:
            top_p_warp = tf.function(top_p_warp, jit_compile=True)
        filtered_dist = tf.exp(top_p_warp(input_ids, dist, cur_len))

        # dist should be filtered to keep min num values so that sum is >= top_p
        # exp (-inf) => 0
        EXPECTED_FILTERED_DIST = tf.constant([[0.3, 0.0, 0.0, 0.5], [0.0, 0.3, 0.3, 0.25]], dtype=tf.float32)
        tf.debugging.assert_near(filtered_dist, EXPECTED_FILTERED_DIST, atol=1e-3)

        # check edge cases with negative and extreme logits
        ramp_logits = np.broadcast_to(
            np.arange(vocab_size, dtype=np.float32)[None, :], (batch_size, vocab_size)
        ).copy() - (vocab_size // 2)

        # make ramp_logits more extreme
        ramp_logits[1] = ramp_logits[1] * 100.0

        # make sure at least 2 tokens are kept
        top_p_warp = TFTopPLogitsWarper(0.9, min_tokens_to_keep=2, filter_value=0.0)
        if use_xla:
            top_p_warp = tf.function(top_p_warp, jit_compile=True)
        filtered_dist = top_p_warp(input_ids, ramp_logits, cur_len)

        # first batch should keep three tokens, second batch would keep only 1, but due to `min_tokens_to_keep=2` keeps
        # 2.
        self.assertListEqual(
            tf.math.reduce_sum(tf.where(filtered_dist != 0.0, 1, 0), axis=-1).numpy().tolist(), [3, 2]
        )

    def test_no_repeat_ngram_dist_processor(self):
        vocab_size = 3
        batch_size = 2
        cur_len = 4

        input_ids = tf.constant([[1, 1, 2, 1], [0, 1, 0, 1]], dtype=tf.int32)
        self.assertEqual(cur_len, input_ids.shape[1])

        scores = self._get_uniform_logits(batch_size, vocab_size)

        no_repeat_proc_2_gram = TFNoRepeatNGramLogitsProcessor(2)
        no_repeat_proc_3_gram = TFNoRepeatNGramLogitsProcessor(3)

        filtered_scores_2_gram = no_repeat_proc_2_gram(input_ids, tf.identity(scores), cur_len)
        filtered_scores_3_gram = no_repeat_proc_3_gram(input_ids, tf.identity(scores), cur_len)

        # 2-gram would forbid 2nd and 3rd token (1,2) at 1st batch and 1st token (0) at 2nd batch
        self.assertListEqual(
            tf.math.is_inf(filtered_scores_2_gram).numpy().tolist(), [[False, True, True], [True, False, False]]
        )

        # 3-gram would forbid no token at 1st batch and 1st token (0) at 2nd batch
        self.assertListEqual(
            tf.math.is_inf(filtered_scores_3_gram).numpy().tolist(), [[False, False, False], [True, False, False]]
        )

    @parameterized.expand([(False,), (True,)])
    def test_no_bad_words_dist_processor(self, use_xla):
        vocab_size = 5
        batch_size = 2
        eos_token_id = 4
        cur_len = 4

        input_ids = tf.constant([[0, 1, 3, 1], [0, 1, 0, 1]], dtype=tf.int32)
        self.assertEqual(cur_len, input_ids.shape[1])

        bad_word_tokens = [[1], [4], [1, 0], [0, 1, 2], [1, 3, 1, 3]]
        scores = self._get_uniform_logits(batch_size, vocab_size)

        no_bad_words_dist_proc = TFNoBadWordsLogitsProcessor(bad_words_ids=bad_word_tokens, eos_token_id=eos_token_id)
        if use_xla:
            no_bad_words_dist_proc = tf.function(no_bad_words_dist_proc, jit_compile=True)

        filtered_scores = no_bad_words_dist_proc(input_ids, tf.identity(scores), cur_len)

        # batch 1: 1st, 2nd, and 4th (0, 1, 3) token are forbidden
        # batch 2: 1st, 2nd, and 3rd (0, 1, 2) token are forbidden
        self.assertListEqual(
            tf.math.is_inf(filtered_scores).numpy().tolist(),
            [[True, True, False, True, True], [True, True, True, False, True]],
        )

    @parameterized.expand([(False,), (True,)])
    def test_forced_bos_token_logits_processor(self, use_xla):
        vocab_size = 20
        batch_size = 4
        bos_token_id = 0

        logits_processor = TFForcedBOSTokenLogitsProcessor(bos_token_id=bos_token_id)
        if use_xla:
            logits_processor = tf.function(logits_processor, jit_compile=True)

        # check that all scores are -inf except the bos_token_id score
        cur_len = 1
        input_ids = ids_tensor((batch_size, cur_len), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores, cur_len)
        self.assertTrue(
            tf.math.reduce_all(tf.math.is_inf(scores[:, bos_token_id + 1 :]) & (scores[:, bos_token_id + 1 :] < 0))
        )
        self.assertListEqual(scores[:, bos_token_id].numpy().tolist(), 4 * [0])  # score for bos_token_id shold be zero

        # check that bos_token_id is not forced if current length is greater than 1
        cur_len = 4
        input_ids = ids_tensor((batch_size, cur_len), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores, cur_len)
        self.assertFalse(tf.math.reduce_any(tf.math.is_inf((scores))))

    @parameterized.expand([(False,), (True,)])
    def test_forced_eos_token_logits_processor(self, use_xla):
        vocab_size = 20
        batch_size = 4
        eos_token_id = 0
        max_length = 5

        logits_processor = TFForcedEOSTokenLogitsProcessor(max_length=max_length, eos_token_id=eos_token_id)
        if use_xla:
            logits_processor = tf.function(logits_processor, jit_compile=True)

        # check that all scores are -inf except the eos_token_id when max_length-1 is reached
        cur_len = 4
        input_ids = ids_tensor((batch_size, cur_len), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores, cur_len)
        self.assertTrue(
            tf.math.reduce_all(tf.math.is_inf(scores[:, eos_token_id + 1 :]) & (scores[:, eos_token_id + 1 :] < 0))
        )
        self.assertListEqual(
            scores[:, eos_token_id].numpy().tolist(), 4 * [0]
        )  # score for eos_token_id should be zero

        # check that eos_token_id is not forced if max_length-1 is not reached
        cur_len = 3
        input_ids = ids_tensor((batch_size, cur_len), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores, cur_len)
        self.assertFalse(tf.math.reduce_any(tf.math.is_inf((scores))))

    @parameterized.expand([(False,), (True,)])
    def test_suppress_tokens_at_begin_logits_processor(self, use_xla):
        vocab_size = 20
        batch_size = 4

        begin_suppress_tokens = [1, 2, 3]
        begin_index = 5

        logits_processor = TFSuppressTokensAtBeginLogitsProcessor(
            begin_suppress_tokens=begin_suppress_tokens, begin_index=begin_index
        )
        if use_xla:
            logits_processor = tf.function(logits_processor, jit_compile=True)

        # Check that no scores are suppressed if begin_index is not reached
        cur_len = 4
        input_ids = tf.convert_to_tensor([[11, 17, 15, 8], [14, 0, 19, 5], [13, 11, 18, 19], [11, 12, 16, 15]])
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores, cur_len)
        self.assertFalse(tf.math.reduce_any(tf.math.is_inf((scores))))

        # Check that scores are suppressed if begin_index is reached
        cur_len = 5
        input_ids = tf.convert_to_tensor([[5, 5, 5, 0, 17], [18, 1, 9, 14, 17], [18, 6, 8, 15, 19], [8, 12, 17, 1, 2]])
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores, cur_len)
        self.assertTrue(tf.math.reduce_all(tf.math.is_inf(tf.gather(scores, begin_suppress_tokens, axis=1))))

    @parameterized.expand([(False,), (True,)])
    def test_suppress_tokens_logits_processor(self, use_xla):
        vocab_size = 20
        batch_size = 4

        suppress_tokens = [1, 3, 5]
        keep_tokens = [i for i in range(vocab_size) if i not in suppress_tokens]

        logits_processor = TFSuppressTokensLogitsProcessor(suppress_tokens=suppress_tokens)
        if use_xla:
            logits_processor = tf.function(logits_processor, jit_compile=True)

        # Check that suppress_tokens are suppressed and others are not
        cur_len = 5
        input_ids = tf.convert_to_tensor([[0, 10, 19, 6, 3], [17, 4, 8, 17, 2], [7, 1, 11, 6, 15], [5, 8, 13, 16, 0]])
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores, cur_len)
        self.assertTrue(tf.math.reduce_all(tf.math.is_inf(tf.gather(scores, suppress_tokens, axis=1))))
        self.assertFalse(tf.math.reduce_any(tf.math.is_inf(tf.gather(scores, keep_tokens, axis=1))))

    @parameterized.expand([(False,), (True,)])
    def test_force_tokens_logits_processor(self, use_xla):
        vocab_size = 20
        batch_size = 4

        force_token_map = {1: 2, 3: 2}

        logits_processor = TFForceTokensLogitsProcessor(force_token_map=force_token_map)
        if use_xla:
            logits_processor = tf.function(logits_processor, jit_compile=True)

        # check that if the cur_len is contained in the force_token_map, the logits are the same
        # for all tokens except the one the force_token_map points to
        cur_len = 1
        input_ids = tf.convert_to_tensor([[11], [7], [5], [15]])
        ids_tensor((batch_size, cur_len), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores, cur_len)
        tf.debugging.assert_near(tf.gather(scores, [force_token_map[cur_len]], axis=1), 0.0)

        non_forced_inds = [i for i in range(vocab_size) if i != force_token_map[cur_len]]
        self.assertTrue(
            tf.math.reduce_all(
                tf.experimental.numpy.isclose(
                    tf.gather(scores, [non_forced_inds], axis=1),
                    tf.constant(scores.dtype.min),
                )
            )
        )

        # check that if the cur_len is not contained in the force_token_map, the logits are not modified
        cur_len = 2
        input_ids = tf.convert_to_tensor([[2, 19], [19, 15], [4, 9], [7, 6]])
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores, cur_len)
        self.assertFalse(tf.math.reduce_any(tf.math.is_inf((scores))))

    @parameterized.expand([(False,), (True,)])
    def test_processor_list(self, use_xla):
        # TODO (Joao): reintroduce TFNoRepeatNGramLogitsProcessor when it gets compatible with XLA
        batch_size = 4
        cur_len = 10
        vocab_size = 15
        eos_token_id = 0

        # dummy input_ids and scores
        input_ids = ids_tensor((batch_size, cur_len), vocab_size)
        input_ids_comp = tf.identity(input_ids)

        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_comp = tf.identity(scores)

        # instantiate all dist processors
        min_dist_proc = TFMinLengthLogitsProcessor(min_length=10, eos_token_id=eos_token_id)
        temp_dist_warp = TFTemperatureLogitsWarper(temperature=0.5)
        rep_penalty_proc = TFRepetitionPenaltyLogitsProcessor(penalty=2.0)
        top_k_warp = TFTopKLogitsWarper(3)
        top_p_warp = TFTopPLogitsWarper(0.8)
        # no_repeat_proc = TFNoRepeatNGramLogitsProcessor(2)
        no_bad_words_dist_proc = TFNoBadWordsLogitsProcessor(bad_words_ids=[[1]], eos_token_id=eos_token_id)
        if use_xla:
            min_dist_proc = tf.function(min_dist_proc, jit_compile=True)
            temp_dist_warp = tf.function(temp_dist_warp, jit_compile=True)
            rep_penalty_proc = tf.function(rep_penalty_proc, jit_compile=True)
            top_k_warp = tf.function(top_k_warp, jit_compile=True)
            top_p_warp = tf.function(top_p_warp, jit_compile=True)
            # no_repeat_proc = tf.function(no_repeat_proc, jit_compile=True)
            no_bad_words_dist_proc = tf.function(no_bad_words_dist_proc, jit_compile=True)

        # no processor list
        scores = min_dist_proc(input_ids, scores, cur_len)
        scores = temp_dist_warp(input_ids, scores, cur_len)
        scores = rep_penalty_proc(input_ids, scores, cur_len)
        scores = top_k_warp(input_ids, scores, cur_len)
        scores = top_p_warp(input_ids, scores, cur_len)
        # scores = no_repeat_proc(input_ids, scores, cur_len)
        scores = no_bad_words_dist_proc(input_ids, scores, cur_len)

        # with processor list
        processor = TFLogitsProcessorList(
            [
                min_dist_proc,
                temp_dist_warp,
                rep_penalty_proc,
                top_k_warp,
                top_p_warp,
                # no_repeat_proc,
                no_bad_words_dist_proc,
            ]
        )
        scores_comp = processor(input_ids, scores_comp, cur_len)

        # remove inf
        scores = tf.where(tf.math.is_inf(scores), -1e9, scores)
        scores_comp = tf.where(tf.math.is_inf(scores_comp), -1e9, scores_comp)

        # scores should be equal
        tf.debugging.assert_near(scores, scores_comp, atol=1e-3)

        # input_ids should never be changed
        self.assertListEqual(input_ids.numpy().tolist(), input_ids_comp.numpy().tolist())
