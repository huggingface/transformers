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

from transformers import is_tf_available
from transformers.testing_utils import require_tf


if is_tf_available():
    import tensorflow as tf

    from transformers.generation_tf_logits_process import (
        TFLogitsProcessorList,
        TFMinLengthLogitsProcessor,
        TFNoBadWordsLogitsProcessor,
        TFNoRepeatNGramLogitsProcessor,
        TFRepetitionPenaltyLogitsProcessor,
    )
    from transformers.tf_utils import set_tensor_by_indices_to_value

    from ..test_modeling_tf_common import ids_tensor


@require_tf
class TFLogitsProcessorTest(unittest.TestCase):
    def _get_uniform_logits(self, batch_size: int, length: int):
        scores = tf.ones((batch_size, length), dtype=tf.float32) / length
        return scores

    def test_min_length_dist_processor(self):
        vocab_size = 20
        batch_size = 4
        eos_token_id = 0

        min_dist_processor = TFMinLengthLogitsProcessor(min_length=10, eos_token_id=eos_token_id)

        # check that min length is applied at length 5
        input_ids = ids_tensor((batch_size, 5), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = min_dist_processor(input_ids, scores)
        self.assertListEqual(scores_before_min_length[:, eos_token_id].numpy().tolist(), 4 * [-float("inf")])

        # check that min length is not applied anymore at length 15
        input_ids = ids_tensor((batch_size, 15), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = min_dist_processor(input_ids, scores)
        self.assertFalse(tf.math.reduce_any(tf.math.is_inf(scores_before_min_length)).numpy())

    def test_repetition_penalty_dist_process(self):
        input_ids = tf.constant([[0, 1], [5, 0]], dtype=tf.int32)
        vocab_size = 10

        scores = self._get_uniform_logits(batch_size=2, length=vocab_size)

        mask = tf.cast(tf.constant([[1] + 9 * [0], 10 * [0]]), tf.bool)
        scores = set_tensor_by_indices_to_value(scores, mask, -1 / vocab_size)
        mask = tf.cast(tf.constant([10 * [0], 5 * [0] + [1] + 4 * [0]]), tf.bool)
        scores = set_tensor_by_indices_to_value(scores, mask, 4 / vocab_size)

        rep_penalty_proc = TFRepetitionPenaltyLogitsProcessor(penalty=2.0)

        scores = rep_penalty_proc(input_ids, tf.identity(scores))

        # check that values were correctly changed
        self.assertAlmostEqual(scores[0, 0].numpy(), -(1 / vocab_size) * 2)
        self.assertAlmostEqual(scores[0, 1].numpy(), (1 / vocab_size) / 2)

        self.assertAlmostEqual(scores[1, 0].numpy(), (1 / vocab_size) / 2)
        self.assertAlmostEqual(scores[1, 5].numpy(), (4 / vocab_size) / 2)

    def test_no_repeat_ngram_dist_processor(self):
        vocab_size = 3
        batch_size = 2

        input_ids = tf.constant([[1, 1, 2, 1], [0, 1, 0, 1]], dtype=tf.int32)
        scores = self._get_uniform_logits(batch_size, vocab_size)

        no_repeat_proc_2_gram = TFNoRepeatNGramLogitsProcessor(2)
        no_repeat_proc_3_gram = TFNoRepeatNGramLogitsProcessor(3)

        filtered_scores_2_gram = no_repeat_proc_2_gram(input_ids, tf.identity(scores))
        filtered_scores_3_gram = no_repeat_proc_3_gram(input_ids, tf.identity(scores))

        # 2-gram would forbid 2nd and 3rd token (1,2) at 1st batch and 1st token (0) at 2nd batch
        self.assertListEqual(
            tf.math.is_inf(filtered_scores_2_gram).numpy().tolist(), [[False, True, True], [True, False, False]]
        )

        # 3-gram would forbid no token at 1st batch and 1st token (0) at 2nd batch
        self.assertListEqual(
            tf.math.is_inf(filtered_scores_3_gram).numpy().tolist(), [[False, False, False], [True, False, False]]
        )

    def test_no_bad_words_dist_processor(self):
        vocab_size = 5
        batch_size = 2
        eos_token_id = 4

        input_ids = tf.constant([[0, 1, 3, 1], [0, 1, 0, 1]], dtype=tf.int32)
        bad_word_tokens = [[1], [4], [1, 0], [0, 1, 2], [1, 3, 1, 3]]
        scores = self._get_uniform_logits(batch_size, vocab_size)

        no_bad_words_dist_proc = TFNoBadWordsLogitsProcessor(bad_words_ids=bad_word_tokens, eos_token_id=eos_token_id)

        filtered_scores = no_bad_words_dist_proc(input_ids, tf.identity(scores))

        # batch 1: 1st, 2nd, and 4th (0, 1, 3) token are forbidden
        # batch 2: 1st, 2nd, and 3rd (0, 1, 2) token are forbidden
        self.assertListEqual(
            tf.math.is_inf(filtered_scores).numpy().tolist(),
            [[True, True, False, True, True], [True, True, True, False, True]],
        )

    def test_processor_list(self):
        batch_size = 4
        sequence_length = 10
        vocab_size = 15
        eos_token_id = 0

        # dummy input_ids and scores
        input_ids = ids_tensor((batch_size, sequence_length), vocab_size)
        input_ids_comp = tf.identity(input_ids)

        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_comp = tf.identity(scores)

        # instantiate all dist processors
        min_dist_proc = TFMinLengthLogitsProcessor(min_length=10, eos_token_id=eos_token_id)
        rep_penalty_proc = TFRepetitionPenaltyLogitsProcessor(penalty=2.0)
        no_repeat_proc = TFNoRepeatNGramLogitsProcessor(2)
        no_bad_words_dist_proc = TFNoBadWordsLogitsProcessor(bad_words_ids=[[1]], eos_token_id=eos_token_id)

        # no processor list
        scores = min_dist_proc(input_ids, scores)
        scores = rep_penalty_proc(input_ids, scores)
        scores = no_repeat_proc(input_ids, scores)
        scores = no_bad_words_dist_proc(input_ids, scores)

        # with processor list
        processor = TFLogitsProcessorList(
            [
                min_dist_proc,
                rep_penalty_proc,
                no_repeat_proc,
                no_bad_words_dist_proc,
            ]
        )
        scores_comp = processor(input_ids, scores_comp)

        # remove inf
        scores = set_tensor_by_indices_to_value(scores, tf.math.is_inf(scores), -1e9)
        scores_comp = set_tensor_by_indices_to_value(scores_comp, tf.math.is_inf(scores_comp), -1e9)

        # scores should be equal
        tf.debugging.assert_near(scores, scores_comp, atol=1e-3)

        # input_ids should never be changed
        self.assertListEqual(input_ids.numpy().tolist(), input_ids_comp.numpy().tolist())
