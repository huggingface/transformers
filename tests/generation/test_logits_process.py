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
from typing import List, Union

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ..test_modeling_common import ids_tensor


if is_torch_available():
    import torch
    from torch import nn

    from transformers.generation import (
        EncoderNoRepeatNGramLogitsProcessor,
        EncoderRepetitionPenaltyLogitsProcessor,
        EpsilonLogitsWarper,
        EtaLogitsWarper,
        ExponentialDecayLengthPenalty,
        ForcedBOSTokenLogitsProcessor,
        ForcedEOSTokenLogitsProcessor,
        HammingDiversityLogitsProcessor,
        InfNanRemoveLogitsProcessor,
        LogitNormalization,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        MinNewTokensLengthLogitsProcessor,
        NoBadWordsLogitsProcessor,
        NoRepeatNGramLogitsProcessor,
        PrefixConstrainedLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        SequenceBiasLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
        TypicalLogitsWarper,
        UnbatchedClassifierFreeGuidanceLogitsProcessor,
    )


@require_torch
class LogitsProcessorTest(unittest.TestCase):
    def _get_uniform_logits(self, batch_size: int, length: int):
        scores = torch.ones((batch_size, length), device=torch_device, dtype=torch.float) / length
        return scores

    def test_min_length_dist_processor(self):
        vocab_size = 20
        batch_size = 4
        eos_token_id = 0

        min_dist_processor = MinLengthLogitsProcessor(min_length=10, eos_token_id=eos_token_id)

        # check that min length is applied at length 5
        input_ids = ids_tensor((batch_size, 5), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = min_dist_processor(input_ids, scores)
        self.assertListEqual(scores_before_min_length[:, eos_token_id].tolist(), 4 * [-float("inf")])

        # check that min length is not applied anymore at length 15
        input_ids = ids_tensor((batch_size, 15), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = min_dist_processor(input_ids, scores)
        self.assertFalse(torch.isinf(scores_before_min_length).any())

    @parameterized.expand([(0,), ([0, 18],)])
    def test_new_min_length_dist_processor(self, eos_token_id: Union[int, List[int]]):
        vocab_size = 20
        batch_size = 4

        # check that first input is skipped (min new length applying)
        input_ids = ids_tensor((batch_size, 5), vocab_size=20)
        new_min_dist_processor = MinNewTokensLengthLogitsProcessor(
            prompt_length_to_skip=input_ids.shape[-1], min_new_tokens=3, eos_token_id=eos_token_id
        )

        expected_eos_scores_before_min_length = batch_size * [-float("inf")]
        if isinstance(eos_token_id, list):
            expected_eos_scores_before_min_length *= len(eos_token_id)

        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = new_min_dist_processor(input_ids, scores)
        self.assertListEqual(
            scores_before_min_length[:, eos_token_id].flatten().tolist(), expected_eos_scores_before_min_length
        )

        # check that, for skipping, now prompt length is 5, after that we expect first 5 tokens will be skipped
        self.assertTrue(new_min_dist_processor.prompt_length_to_skip == 5)

        # check that min length is applied at length 2
        input_ids = ids_tensor((batch_size, 2), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = new_min_dist_processor(input_ids, scores)
        self.assertListEqual(
            scores_before_min_length[:, eos_token_id].flatten().tolist(), expected_eos_scores_before_min_length
        )

        # check that min new length is applied at length 6 (because it has only 1 new token)
        input_ids = ids_tensor((batch_size, 6), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = new_min_dist_processor(input_ids, scores)
        self.assertListEqual(
            scores_before_min_length[:, eos_token_id].flatten().tolist(), expected_eos_scores_before_min_length
        )

        # check that min new length is applied at length 7 (because it has only 2 new tokens)
        input_ids = ids_tensor((batch_size, 7), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = new_min_dist_processor(input_ids, scores)
        self.assertListEqual(
            scores_before_min_length[:, eos_token_id].flatten().tolist(), expected_eos_scores_before_min_length
        )

        # check that min new length is not applied anymore at length 8
        input_ids = ids_tensor((batch_size, 8), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = new_min_dist_processor(input_ids, scores)
        self.assertFalse(torch.isinf(scores_before_min_length).any())

        # check that min new length is not applied anymore at length 15
        input_ids = ids_tensor((batch_size, 15), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = new_min_dist_processor(input_ids, scores)
        self.assertFalse(torch.isinf(scores_before_min_length).any())

    def test_temperature_dist_warper(self):
        input_ids = None
        length = 20

        scores = self._get_uniform_logits(batch_size=2, length=length)

        # tweak scores to not be uniform anymore
        scores[1, 5] = (1 / length) + 0.1  # peak, 1st batch
        scores[1, 10] = (1 / length) - 0.4  # valley, 1st batch

        # compute softmax
        probs = nn.functional.softmax(scores, dim=-1)

        temp_dist_warper_sharper = TemperatureLogitsWarper(temperature=0.5)
        temp_dist_warper_smoother = TemperatureLogitsWarper(temperature=1.3)

        warped_prob_sharp = nn.functional.softmax(temp_dist_warper_sharper(input_ids, scores.clone()), dim=-1)
        warped_prob_smooth = nn.functional.softmax(temp_dist_warper_smoother(input_ids, scores.clone()), dim=-1)

        # uniform distribution stays uniform
        self.assertTrue(torch.allclose(probs[0, :], warped_prob_sharp[0, :], atol=1e-3))
        self.assertTrue(torch.allclose(probs[0, :], warped_prob_smooth[0, :], atol=1e-3))

        # sharp peaks get higher, valleys get lower
        self.assertLess(probs[1, :].max(), warped_prob_sharp[1, :].max())
        self.assertGreater(probs[1, :].min(), warped_prob_sharp[1, :].min())

        # smooth peaks get lower, valleys get higher
        self.assertGreater(probs[1, :].max(), warped_prob_smooth[1, :].max())
        self.assertLess(probs[1, :].min(), warped_prob_smooth[1, :].min())

    def test_repetition_penalty_dist_process(self):
        input_ids = torch.tensor([[0, 1], [5, 0]], device=torch_device, dtype=torch.long)
        vocab_size = 10

        scores = self._get_uniform_logits(batch_size=2, length=vocab_size)

        # give values special values
        scores[0, 0] = -(1 / vocab_size)
        scores[1, 5] = 4 / vocab_size

        rep_penalty_proc = RepetitionPenaltyLogitsProcessor(penalty=2.0)

        scores = rep_penalty_proc(input_ids, scores.clone())

        # check that values were correctly changed
        self.assertAlmostEqual(scores[0, 0].item(), -(1 / vocab_size) * 2)
        self.assertAlmostEqual(scores[0, 1].item(), (1 / vocab_size) / 2)

        self.assertAlmostEqual(scores[1, 0].item(), (1 / vocab_size) / 2)
        self.assertAlmostEqual(scores[1, 5].item(), (4 / vocab_size) / 2)

    def test_encoder_repetition_penalty_dist_process(self):
        input_ids = torch.tensor([[0, 1], [5, 0]], device=torch_device, dtype=torch.long)
        vocab_size = 10

        scores = self._get_uniform_logits(batch_size=2, length=vocab_size)

        # give values special values
        scores[0, 0] = -(1 / vocab_size)
        scores[1, 5] = 4 / vocab_size

        rep_penalty_proc = EncoderRepetitionPenaltyLogitsProcessor(penalty=2.0, encoder_input_ids=input_ids)

        scores = rep_penalty_proc(input_ids, scores.clone())

        # check that values were correctly changed
        self.assertAlmostEqual(scores[0, 0].item(), -(1 / vocab_size) / 2)
        self.assertAlmostEqual(scores[0, 1].item(), (1 / vocab_size) * 2)

        self.assertAlmostEqual(scores[1, 0].item(), (1 / vocab_size) * 2)
        self.assertAlmostEqual(scores[1, 5].item(), (4 / vocab_size) * 2)

        # check that values not in the encoder ids were NOT changed
        self.assertAlmostEqual(scores[0, 2].item(), (1 / vocab_size))
        self.assertAlmostEqual(scores[1, 2].item(), (1 / vocab_size))

    def test_top_k_dist_warper(self):
        input_ids = None
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

        top_p_warp = TopPLogitsWarper(0.8)
        filtered_dist = torch.exp(top_p_warp(input_ids, dist))

        # dist should be filtered to keep min num values so that sum is >= top_p
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

    def test_typical_dist_warper(self):
        input_ids = None
        vocab_size = 10
        batch_size = 2

        # create distribution and take log (inverse to Softmax as taken in TopPLogitsWarper)
        dist = torch.log(
            torch.tensor([[0.97, 0.01, 0.01, 0.01], [0.4, 0.2, 0.2, 0.2]], device=torch_device, dtype=torch.float)
        )

        typical_warp = TypicalLogitsWarper(0.5)
        filtered_dist = torch.exp(typical_warp(input_ids, dist))

        # dist should be filtered to keep min num values so that sum is >= 0.7
        # exp (-inf) => 0
        EXPECTED_FILTERED_DIST = torch.tensor(
            [[0.97, 0.0, 0.0, 0.0], [0.0, 0.2, 0.2, 0.2]], device=torch_device, dtype=torch.float
        )
        self.assertTrue(torch.allclose(filtered_dist, EXPECTED_FILTERED_DIST, atol=1e-3))

        # check special cases
        length = 5

        logits = self._get_uniform_logits(batch_size=batch_size, length=length)
        typical_warp_safety_check = TypicalLogitsWarper(mass=0.5, filter_value=0.0, min_tokens_to_keep=3)

        scores = typical_warp_safety_check(input_ids, logits)
        # uniform dist is not changed
        self.assertListEqual((scores == 0.0).to(torch.long).sum(dim=-1).tolist(), [0, 0])

        # check edge cases with negative and extreme logits
        ramp_logits = torch.arange(vocab_size, device=torch_device, dtype=torch.float).unsqueeze(0).repeat(
            batch_size, 1
        ) - (vocab_size // 2)

        # make ramp_logits more extreme
        ramp_logits[1] = ramp_logits[1] * 100.0

        # make sure at least 2 tokens are kept
        typical_warp = TypicalLogitsWarper(0.7, min_tokens_to_keep=2, filter_value=0.0)
        filtered_dist = typical_warp(input_ids, ramp_logits)

        # first batch should keep two tokens, second batch would keep only 1, but due to `min_tokens_to_keep=2` keeps 2.
        self.assertListEqual((filtered_dist != 0.0).to(torch.long).sum(dim=-1).tolist(), [2, 2])

    def test_epsilon_dist_warper(self):
        input_ids = None
        vocab_size = 10
        batch_size = 2

        # create distribution and take log (inverse to Softmax as taken in TopPLogitsWarper)
        dist = torch.log(
            torch.tensor(
                [[0.87, 0.099, 0.001, 0.03], [0.4, 0.299, 0.101, 0.2]], device=torch_device, dtype=torch.float
            )
        )

        epsilon_warp = EpsilonLogitsWarper(0.1)
        filtered_dist = torch.exp(epsilon_warp(input_ids, dist))

        # dist should be filtered to only keep values with proba >= 0.1
        # exp (-inf) => 0
        EXPECTED_FILTERED_DIST = torch.tensor(
            [[0.87, 0, 0, 0], [0.4, 0.299, 0.101, 0.2]], device=torch_device, dtype=torch.float
        )
        self.assertTrue(torch.allclose(filtered_dist, EXPECTED_FILTERED_DIST, atol=1e-3))

        # check edge cases with negative and extreme logits
        ramp_logits = torch.arange(vocab_size, device=torch_device, dtype=torch.float).unsqueeze(0).repeat(
            batch_size, 1
        ) - (vocab_size // 2)

        # make ramp_logits more extreme
        ramp_logits[1] = ramp_logits[1] * 100.0

        # make sure at least 2 tokens are kept
        epsilon_warp = EpsilonLogitsWarper(5e-2, min_tokens_to_keep=2, filter_value=0.0)
        filtered_dist = epsilon_warp(input_ids, ramp_logits)

        # first batch should keep 3 tokens, second batch would keep only 1, but due to `min_tokens_to_keep=2` keeps 2.
        self.assertListEqual((filtered_dist != 0.0).to(torch.long).sum(dim=-1).tolist(), [3, 2])

    def test_eta_dist_warper(self):
        input_ids = None
        vocab_size = 10
        batch_size = 2

        # create distribution and take log (inverse to Softmax as taken in TopPLogitsWarper)
        dist = torch.log(
            torch.tensor([[0.0, 0.1, 0.8, 0.1], [0.01, 0.04, 0.9, 0.05]], device=torch_device, dtype=torch.float)
        )

        eta_warp = EtaLogitsWarper(0.0625)
        filtered_dist = torch.exp(eta_warp(input_ids, dist))

        # dist should be filtered to only keep values with proba >= min(0.0625, sqrt(0.0625) * e^-H(p))
        # min(0.0625, 0.1320) is the cutoff for the first row and min(0.0625, 0.1644) is for the second
        # where H is the entropy function and p is the probability vector.
        # exp (-inf) => 0
        EXPECTED_FILTERED_DIST = torch.tensor(
            [[0.0, 0.1, 0.8, 0.1], [0.0, 0.0, 0.9, 0.0]], device=torch_device, dtype=torch.float
        )
        self.assertTrue(torch.allclose(filtered_dist, EXPECTED_FILTERED_DIST, atol=1e-3))

        # check edge cases with negative and extreme logits
        ramp_logits = torch.arange(vocab_size, device=torch_device, dtype=torch.float).unsqueeze(0).repeat(
            batch_size, 1
        ) - (vocab_size // 2)

        # make ramp_logits more extreme
        ramp_logits[1] = ramp_logits[1] * 100.0

        # make sure at least 2 tokens are kept
        eta_warp = EtaLogitsWarper(0.1, min_tokens_to_keep=2, filter_value=0.0)
        filtered_dist = eta_warp(input_ids, ramp_logits)

        # first batch should keep 2 tokens, second batch would keep only 1, but due to `min_tokens_to_keep=2` keeps 2.
        self.assertListEqual((filtered_dist != 0.0).to(torch.long).sum(dim=-1).tolist(), [2, 2])

    def test_no_repeat_ngram_dist_processor(self):
        vocab_size = 3
        batch_size = 2

        input_ids = torch.tensor([[1, 1, 2, 1], [0, 1, 0, 1]], device=torch_device, dtype=torch.long)
        scores = self._get_uniform_logits(batch_size, vocab_size)

        no_repeat_proc_2_gram = NoRepeatNGramLogitsProcessor(2)
        no_repeat_proc_3_gram = NoRepeatNGramLogitsProcessor(3)

        filtered_scores_2_gram = no_repeat_proc_2_gram(input_ids, scores.clone())
        filtered_scores_3_gram = no_repeat_proc_3_gram(input_ids, scores.clone())

        # 2-gram would forbid 2nd and 3rd token (1,2) at 1st batch and 1st token (0) at 2nd batch
        self.assertListEqual(torch.isinf(filtered_scores_2_gram).tolist(), [[False, True, True], [True, False, False]])

        # 3-gram would forbid no token at 1st batch and 1st token (0) at 2nd batch
        self.assertListEqual(
            torch.isinf(filtered_scores_3_gram).tolist(), [[False, False, False], [True, False, False]]
        )

    def test_encoder_no_repeat_ngram_dist_processor(self):
        vocab_size = 3
        num_beams = 2
        batch_size = 1

        encoder_input_ids = torch.tensor([1, 2, 1, 1], device=torch_device, dtype=torch.long)

        input_ids = torch.tensor([[1, 2, 1], [8, 0, 2]], device=torch_device, dtype=torch.long)
        scores = self._get_uniform_logits(batch_size * num_beams, vocab_size)

        no_repeat_proc_2_gram = EncoderNoRepeatNGramLogitsProcessor(2, encoder_input_ids=encoder_input_ids)
        no_repeat_proc_3_gram = EncoderNoRepeatNGramLogitsProcessor(3, encoder_input_ids=encoder_input_ids)

        filtered_scores_2_gram = no_repeat_proc_2_gram(input_ids, scores.clone())
        filtered_scores_3_gram = no_repeat_proc_3_gram(input_ids, scores.clone())

        # 2-gram would forbid 1st and 2nd token at 1st beam and 1st token (0) at 2nd beam
        self.assertListEqual(torch.isinf(filtered_scores_2_gram).tolist(), [[False, True, True], [False, True, False]])

        # 3-gram would forbid 1st token at 1st beam and no token at 2nd beam
        self.assertListEqual(
            torch.isinf(filtered_scores_3_gram).tolist(), [[False, True, False], [False, False, False]]
        )

        # Batched input
        vocab_size = 3
        num_beams = 2
        batch_size = 2
        encoder_input_ids = torch.tensor([[1, 2, 1, 1], [0, 0, 2, 1]], device=torch_device, dtype=torch.long)

        input_ids = torch.tensor([[1, 2, 1], [1, 0, 2], [0, 0, 0], [0, 2, 2]], device=torch_device, dtype=torch.long)
        scores = self._get_uniform_logits(batch_size * num_beams, vocab_size)

        no_repeat_proc_2_gram = EncoderNoRepeatNGramLogitsProcessor(2, encoder_input_ids=encoder_input_ids)
        no_repeat_proc_3_gram = EncoderNoRepeatNGramLogitsProcessor(3, encoder_input_ids=encoder_input_ids)

        filtered_scores_2_gram = no_repeat_proc_2_gram(input_ids, scores.clone())
        filtered_scores_3_gram = no_repeat_proc_3_gram(input_ids, scores.clone())

        # 2gram
        # Batch 1
        #   - Beam 1: tokens (1, 2) forbidden
        #   - Beam 2: tokens (1) forbidden
        # Batch 2
        #   - Beam 1: tokens (0, 2) forbidden
        #   - Beam 2: tokens (1) forbidden
        self.assertListEqual(
            torch.isinf(filtered_scores_2_gram).tolist(),
            [[False, True, True], [False, True, False], [True, False, True], [False, True, False]],
        )

        # Batch 1
        #   - Beam 1: tokens (1) forbidden
        #   - Beam 2: tokens () forbidden
        # Batch 2
        #   - Beam 1: tokens (2) forbidden
        #   - Beam 2: tokens () forbidden
        self.assertListEqual(
            torch.isinf(filtered_scores_3_gram).tolist(),
            [[False, True, False], [False, False, False], [False, False, True], [False, False, False]],
        )

    def test_no_bad_words_dist_processor(self):
        vocab_size = 5
        batch_size = 2
        eos_token_id = 4

        input_ids = torch.tensor([[0, 1, 3, 1], [0, 1, 0, 1]], device=torch_device, dtype=torch.long)
        bad_word_tokens = [[1], [4], [1, 0], [0, 1, 2], [1, 3, 1, 3]]
        scores = self._get_uniform_logits(batch_size, vocab_size)

        no_bad_words_dist_proc = NoBadWordsLogitsProcessor(bad_words_ids=bad_word_tokens, eos_token_id=eos_token_id)

        filtered_scores = no_bad_words_dist_proc(input_ids, scores.clone())

        # batch 1: 1st, 2nd, and 4th (0, 1, 3) token are forbidden
        # batch 2: 1st, 2nd, and 3rd (0, 1, 2) token are forbidden
        # Note that 5th element cannot be forbidden as it is EOS token
        self.assertListEqual(
            torch.isinf(filtered_scores).tolist(), [[True, True, False, True, False], [True, True, True, False, False]]
        )

        # check edge case
        no_bad_words_dist_proc = NoBadWordsLogitsProcessor(bad_words_ids=[[4]], eos_token_id=eos_token_id)
        filtered_scores = no_bad_words_dist_proc(input_ids, scores.clone())
        self.assertTrue(torch.allclose(scores, filtered_scores, atol=1e-3))

    def test_bias_dist_processor(self):
        vocab_size = 5
        batch_size = 2

        input_ids = torch.tensor([[0, 1, 3, 1], [0, 1, 0, 1]], device=torch_device, dtype=torch.long)
        positive_bias = {(1,): 100.0, (4,): 100.0}
        negative_bias = {(1, 0): -100.0, (0, 1, 2): -100.0, (1, 3, 1, 3): -100.0}
        # biases the same termination twice, to ensure we can handle overlapping terminations (it won't have an effect
        # on the test cases, though)
        negative_bias.update({(1, 3, 1, 3, 1, 3): -100.0})
        sequence_bias = {**positive_bias, **negative_bias}

        # scores = 0 to facilitate checks
        scores = torch.zeros((batch_size, vocab_size), dtype=torch.float, device=torch_device)

        bias_dist_proc = SequenceBiasLogitsProcessor(sequence_bias=sequence_bias)
        filtered_scores = bias_dist_proc(input_ids, scores.clone())

        # batch 1: positive bias: tokens (1, 4); negative bias: tokens (0, 3); neutral: tokens (2)
        # batch 2: positive bias: tokens (1, 4); negative bias: tokens (0, 2); neutral: tokens (3)
        self.assertListEqual(
            filtered_scores.tolist(), [[-100.0, 100.0, 0.0, -100.0, 100.0], [-100.0, 100.0, -100.0, 0.0, 100.0]]
        )

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

    def test_prefix_constrained_logits_processor(self):
        vocab_size = 5
        batch_size = 2

        input_ids = torch.tensor([[0, 1, 3, 1], [0, 1, 0, 1]], device=torch_device, dtype=torch.long)
        scores = self._get_uniform_logits(batch_size, vocab_size)

        def prefix_allowed_tokens_fn(batch_id, inputs_ids):
            return [[0, 1], [2, 3]][batch_id]

        prefix_constrained_logits_proc = PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, 1)

        filtered_scores = prefix_constrained_logits_proc(input_ids, scores.clone())

        # batch 1: 1st, 2nd (0, 1) token are allowed
        # batch 2: 3rd, 4th (2, 3) token are allowed
        self.assertListEqual(
            torch.isinf(filtered_scores).tolist(), [[False, False, True, True, True], [True, True, False, False, True]]
        )

    def test_hamming_diversity(self):
        vocab_size = 4
        num_beams = 2
        num_beam_groups = 2

        scores = self._get_uniform_logits(num_beams, vocab_size)
        # batch_idx = 0 -> index batch_idx * num_beam_groups -> idx = 0 * 2 = 0 -> penalises tokens 1
        # batch_idx = 1 -> index batch_idx * num_beam_groups -> idx = 1 * 2 = 2 -> penalises tokens 1
        current_tokens = torch.tensor([0, 3, 1, 2], device=torch_device, dtype=torch.long)

        diversity_logits_processor = HammingDiversityLogitsProcessor(
            diversity_penalty=1.0, num_beams=num_beams, num_beam_groups=num_beam_groups
        )

        processed_scores = diversity_logits_processor(None, scores, current_tokens, 1)

        self.assertTrue(
            torch.allclose(
                processed_scores[0], torch.tensor([-0.7500, 0.2500, 0.2500, 0.2500], device=torch_device), atol=1e-3
            )
        )
        self.assertTrue(
            torch.allclose(
                processed_scores[1], torch.tensor([0.2500, -0.7500, 0.2500, 0.2500], device=torch_device), atol=1e-3
            )
        )

    def test_forced_bos_token_logits_processor(self):
        vocab_size = 20
        batch_size = 4
        bos_token_id = 0

        logits_processor = ForcedBOSTokenLogitsProcessor(bos_token_id=bos_token_id)

        # check that all scores are -inf except the bos_token_id score
        input_ids = ids_tensor((batch_size, 1), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores)
        self.assertTrue(torch.isneginf(scores[:, bos_token_id + 1 :]).all())
        self.assertListEqual(scores[:, bos_token_id].tolist(), 4 * [0])  # score for bos_token_id shold be zero

        # check that bos_token_id is not forced if current length is greater than 1
        input_ids = ids_tensor((batch_size, 4), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores)
        self.assertFalse(torch.isinf(scores).any())

    def test_forced_eos_token_logits_processor(self):
        vocab_size = 20
        batch_size = 4
        eos_token_id = 0
        max_length = 5

        logits_processor = ForcedEOSTokenLogitsProcessor(max_length=max_length, eos_token_id=eos_token_id)

        # check that all scores are -inf except the eos_token_id when max_length-1 is reached
        input_ids = ids_tensor((batch_size, 4), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores)
        self.assertTrue(torch.isneginf(scores[:, eos_token_id + 1 :]).all())
        self.assertListEqual(scores[:, eos_token_id].tolist(), 4 * [0])  # score for eos_token_id should be zero

        # check that eos_token_id is not forced if max_length-1 is not reached
        input_ids = ids_tensor((batch_size, 3), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores)
        self.assertFalse(torch.isinf(scores).any())

    def test_remove_nan_inf_logits_processor(self):
        scores = torch.tensor(
            [[0.0, 0.7, 0.8, float("nan")], [0.1, float("inf"), 0.3, float("-inf")]], device=torch_device
        )
        input_ids = ids_tensor((2, 4), vocab_size=20)

        logits_processor = InfNanRemoveLogitsProcessor()

        scores = logits_processor(input_ids, scores)

        self.assertTrue(
            torch.allclose(
                scores,
                torch.tensor(
                    [[0.0, 0.7, 0.8, 0.0], [0.1, torch.finfo(scores.dtype).max, 0.3, float("-inf")]],
                    device=torch_device,
                ),
                atol=1e-6,
            )
        )

    def test_exponential_decay_length_penalty(self):
        vocab_size = 20
        batch_size = 4
        eos_token_id = 0

        penalty_start = 5
        penalty_factor = 1.1

        input_ids = ids_tensor((batch_size, 2), vocab_size=vocab_size)
        input_ids_seq_length = input_ids.shape[-1]

        length_decay_processor = ExponentialDecayLengthPenalty(
            exponential_decay_length_penalty=(penalty_start, penalty_factor),
            eos_token_id=eos_token_id,
            input_ids_seq_length=input_ids_seq_length,
        )

        # check that penalty is not applied before start
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_start = torch.clone(scores)  # clone scores as precessor updates them inplace
        scores_before_start = length_decay_processor(input_ids, scores_before_start)
        self.assertListEqual(scores_before_start[:, eos_token_id].tolist(), scores[:, eos_token_id].tolist())

        # check that penalty is applied after start
        input_ids = ids_tensor((batch_size, 20), vocab_size=vocab_size)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_after_start = torch.clone(scores)  # clone scores as precessor updates them inplace
        scores_after_start = length_decay_processor(input_ids, scores_after_start)
        self.assertTrue(torch.gt(scores_after_start[:, eos_token_id], scores[:, eos_token_id]).all())

        # check the penalty increases negative scores
        input_ids = ids_tensor((batch_size, 20), vocab_size=vocab_size)
        scores = torch.neg(self._get_uniform_logits(batch_size, vocab_size))
        scores_after_start = torch.clone(scores)  # clone scores as precessor updates them inplace
        scores_after_start = length_decay_processor(input_ids, scores_after_start)
        self.assertTrue(torch.gt(scores_after_start[:, eos_token_id], scores[:, eos_token_id]).all())

    def test_normalization(self):
        input_ids = None

        scores = torch.tensor(
            [[-23.18, -29.96, -43.54, 47.77], [-33.58, -26.87, -32.96, 22.51]], device=torch_device, dtype=torch.float
        )

        logit_normalization = LogitNormalization()
        normalized_scores = logit_normalization(input_ids, scores).exp()

        ones = torch.ones(scores.shape[0], device=torch_device, dtype=torch.float)
        self.assertTrue(normalized_scores.sum(dim=-1).allclose(ones))

        self.assertTrue(normalized_scores.allclose(scores.softmax(dim=-1)))

    def test_classifier_free_guidance(self):
        class Namespace(dict):
            pass

        logits_uncond = torch.tensor([[[1.0, 0, 1.5]]])
        logits_cond = torch.tensor([[[1.0, 1.0, 1.0]]])

        def dummy_model(input_ids, attention_mask, use_cache=True, past_key_values=None):
            out = Namespace()
            out.logits = logits_uncond
            out.past_key_values = None
            return out

        def lsm(x):
            return torch.nn.functional.log_softmax(x, dim=-1)

        # explicit unconditional prompt + attention mask
        input_ids = torch.LongTensor([[0]])
        cfg = UnbatchedClassifierFreeGuidanceLogitsProcessor(
            1.5, dummy_model, input_ids, torch.ones_like(input_ids, dtype=torch.long)
        )
        out = cfg(input_ids, logits_cond)[0, -1]

        res = (lsm(logits_uncond) + 1.5 * (lsm(logits_cond) - lsm(logits_uncond)))[0, -1]

        self.assertAlmostEqual(out[0].item(), res[0].item())
        self.assertAlmostEqual(out[1].item(), res[1].item())
        self.assertAlmostEqual(out[2].item(), res[2].item())

        # explicit unconditional prompt
        input_ids = torch.LongTensor([[0]])
        cfg = UnbatchedClassifierFreeGuidanceLogitsProcessor(1.5, dummy_model, input_ids)
        out = cfg(input_ids, logits_cond)[0, -1]

        res = (lsm(logits_uncond) + 1.5 * (lsm(logits_cond) - lsm(logits_uncond)))[0, -1]

        self.assertAlmostEqual(out[0].item(), res[0].item())
        self.assertAlmostEqual(out[1].item(), res[1].item())
        self.assertAlmostEqual(out[2].item(), res[2].item())

        # all implicit
        input_ids = torch.LongTensor([[0]])
        cfg = UnbatchedClassifierFreeGuidanceLogitsProcessor(1.5, dummy_model)
        out = cfg(input_ids, logits_cond)[0, -1]

        res = (lsm(logits_uncond) + 1.5 * (lsm(logits_cond) - lsm(logits_uncond)))[0, -1]

        self.assertAlmostEqual(out[0].item(), res[0].item())
        self.assertAlmostEqual(out[1].item(), res[1].item())
        self.assertAlmostEqual(out[2].item(), res[2].item())
