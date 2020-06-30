# coding=utf-8
# Copyright 2020 HuggingFace Inc.
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

import unittest

from transformers import is_tf_available, is_torch_available
from transformers.generation_utils_samplers import (
    CompositionSampler,
    IdentitySampler,
    MinLengthSampler,
    NoBadWordsSampler,
    NoRepeatNGramSampler,
    RepetitionPenaltySampler,
    TemperatureSampler,
    TopKSampler,
    TopPSampler,
)

from .utils import require_tf, require_torch, torch_device


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf


class SamplingUnitTests(unittest.TestCase):
    @require_torch
    def test_identity_torch(self):
        sampler = IdentitySampler()
        self.assertEqual(sampler.warp_torch(torch.LongTensor([1]), torch.Tensor([2])), torch.Tensor([2]))

    @require_tf
    def test_identity_tf(self):
        sampler = IdentitySampler()
        self.assertEqual(
            sampler.warp_tf(tf.convert_to_tensor([1]), tf.convert_to_tensor([2])), tf.convert_to_tensor([2])
        )

    @require_torch
    def test_min_length_sampler_torch(self):
        hundred_min_length = MinLengthSampler(min_length=100, eos_token_id=1)
        warped = hundred_min_length.warp_torch(torch.LongTensor([[1]]), torch.Tensor([[10.0, 10.0]]))
        self.assertTrue(torch.all(torch.eq(warped, torch.Tensor([[10.0, -float("inf")]]))))

    @require_tf
    def test_min_length_sampler_tf(self):
        hundred_min_length = MinLengthSampler(min_length=100, eos_token_id=1)
        warped = hundred_min_length.warp_tf(tf.convert_to_tensor([[1]]), tf.convert_to_tensor([10.0, 10.0]))

        tf.debugging.assert_equal(warped, tf.convert_to_tensor([10.0, -float("inf")]))

    @require_torch
    def test_temperature_sampler_torch(self):
        ts = TemperatureSampler(2)
        warped = ts.warp_torch(torch.LongTensor([1]), torch.Tensor([10.0, 5.0]))
        self.assertTrue(torch.allclose(warped, torch.Tensor([[5.0, 2.5]]), atol=1e-12))

    @require_tf
    def test_temperature_sampler_tf(self):
        ts = TemperatureSampler(2)
        warped = ts.warp_tf(tf.convert_to_tensor([1]), tf.convert_to_tensor([10.0, 5.0]))
        tf.debugging.assert_near(warped, tf.convert_to_tensor([[5.0, 2.5]]), rtol=1e-12)

    @require_torch
    def test_repetition_penalty_sampler_torch(self):
        rp = RepetitionPenaltySampler(2.0)
        warped = rp.warp_torch(torch.LongTensor([[1]]), torch.Tensor([[10, 10, 10]]))
        self.assertTrue(torch.allclose(warped, torch.Tensor([[10, 5, 10]]), atol=1e-12))

    @require_tf
    def test_repetition_penalty_sampler_tf(self):
        rp = RepetitionPenaltySampler(2.0)
        warped = rp.warp_tf(tf.convert_to_tensor([[1]], dtype=tf.int32), tf.convert_to_tensor([[10.0, 10.0, 10.0]]))
        tf.debugging.assert_near(warped, tf.convert_to_tensor([[10.0, 5.0, 10.0]]), rtol=1e-12)

    @require_torch
    def test_no_repeat_ngram_sampler_torch(self):
        nr = NoRepeatNGramSampler(2)
        warped = nr.warp_torch(torch.LongTensor([[1, 2, 1]]), torch.Tensor([[10, 10, 10]]))
        self.assertTrue(torch.allclose(warped, torch.Tensor([[10, 10, -float("inf")]]), atol=1e-12))

    @require_tf
    def test_no_repeat_ngram_sampler_tf(self):
        nr = NoRepeatNGramSampler(2)
        warped = nr.warp_tf(
            tf.convert_to_tensor([[1, 2, 1]], dtype=tf.int32), tf.convert_to_tensor([[10.0, 10.0, 10.0]])
        )
        tf.debugging.assert_equal(warped, tf.convert_to_tensor([[10.0, 10.0, -float("inf")]]))

    @require_torch
    def test_bad_words_torch(self):
        nb = NoBadWordsSampler([[1, 2]])
        warped = nb.warp_torch(torch.LongTensor([[1]]), torch.Tensor([[10, 10, 10]]))
        self.assertTrue(torch.allclose(warped, torch.Tensor([[10, 10, -float("inf")]]), atol=1e-12))

    @require_tf
    def test_bad_words_tf(self):
        nr = NoBadWordsSampler([[1, 2]])
        warped = nr.warp_tf(tf.convert_to_tensor([[1]], dtype=tf.int32), tf.convert_to_tensor([[10.0, 10.0, 10.0]]))
        tf.debugging.assert_equal(warped, tf.convert_to_tensor([[10.0, 10.0, -float("inf")]]))

    @require_torch
    def test_top_k_top_p_filtering_torch(self):
        sampler = CompositionSampler(
            (TopKSampler(k=10, min_tokens_to_keep=4), TopPSampler(p=0.6, min_tokens_to_keep=4),)
        )

        # tests whether the top_k_top_p function behaves as expected
        logits = torch.tensor(
            [
                [
                    8.2220991,  # 3rd highest value; idx. 0
                    -0.5620044,
                    5.23229752,
                    4.0386393,
                    -6.8798378,
                    -0.54785802,
                    -3.2012153,
                    2.92777176,
                    1.88171953,
                    7.35341276,  # 5th highest value; idx. 9
                    8.43207833,  # 2nd highest value; idx. 10
                    -9.85711836,
                    -5.96209236,
                    -1.13039161,
                    -7.1115294,
                    -0.8369633,
                    -5.3186408,
                    7.06427407,
                    0.81369344,
                    -0.82023817,
                    -5.9179796,
                    0.58813443,
                    -6.99778438,
                    4.71551189,
                    -0.18771637,
                    7.44020759,  # 4th highest value; idx. 25
                    9.38450987,  # 1st highest value; idx. 26
                    2.12662941,
                    -9.32562038,
                    2.35652522,
                ],  # cummulative prob of 5 highest values <= 0.6
                [
                    0.58425518,
                    4.53139238,
                    -5.57510464,
                    -6.28030699,
                    -7.19529503,
                    -4.02122551,
                    1.39337037,
                    -6.06707057,
                    1.59480517,
                    -9.643119,
                    0.03907799,
                    0.67231762,
                    -8.88206726,
                    6.27115922,  # 4th highest value; idx. 13
                    2.28520723,
                    4.82767506,
                    4.30421368,
                    8.8275313,  # 2nd highest value; idx. 17
                    5.44029958,  # 5th highest value; idx. 18
                    -4.4735794,
                    7.38579536,  # 3rd highest value; idx. 20
                    -2.91051663,
                    2.61946077,
                    -2.5674762,
                    -9.48959302,
                    -4.02922645,
                    -1.35416918,
                    9.67702323,  # 1st highest value; idx. 27
                    -5.89478553,
                    1.85370467,
                ],  # cummulative prob of 5 highest values <= 0.6
            ],
            dtype=torch.float,
            device=torch_device,
        )

        non_inf_expected_idx = torch.tensor(
            [[0, 0], [0, 9], [0, 10], [0, 25], [0, 26], [1, 13], [1, 17], [1, 18], [1, 20], [1, 27]],
            dtype=torch.long,
            device=torch_device,
        )  # expected non filtered idx as noted above

        non_inf_expected_output = torch.tensor(
            [
                8.2221,
                7.3534,
                8.4321,
                7.4402,
                9.3845,
                6.2712,
                8.8275,
                5.4403,
                7.3858,
                9.6770,
            ],  # expected non filtered values as noted above
            dtype=torch.float,
            device=torch_device,
        )

        output = sampler.warp_torch([], logits)
        non_inf_output = output[output != -float("inf")].to(device=torch_device)
        non_inf_idx = (output != -float("inf")).nonzero().to(device=torch_device)

        self.assertTrue(torch.allclose(non_inf_expected_output, non_inf_output, atol=1e-12))
        self.assertTrue(torch.all(torch.eq(non_inf_expected_idx, non_inf_idx)))

    @require_tf
    def test_top_k_top_p_filtering_tf(self):
        sampler = CompositionSampler(
            (TopKSampler(k=10, min_tokens_to_keep=4), TopPSampler(p=0.6, min_tokens_to_keep=4),)
        )
        # tests whether the top_k_top_p_filtering function behaves as expected
        logits = tf.convert_to_tensor(
            [
                [
                    8.2220991,  # 3rd highest value; idx. 0
                    -0.5620044,
                    5.23229752,
                    4.0386393,
                    -6.8798378,
                    -0.54785802,
                    -3.2012153,
                    2.92777176,
                    1.88171953,
                    7.35341276,  # 5th highest value; idx. 9
                    8.43207833,  # 2nd highest value; idx. 10
                    -9.85711836,
                    -5.96209236,
                    -1.13039161,
                    -7.1115294,
                    -0.8369633,
                    -5.3186408,
                    7.06427407,
                    0.81369344,
                    -0.82023817,
                    -5.9179796,
                    0.58813443,
                    -6.99778438,
                    4.71551189,
                    -0.18771637,
                    7.44020759,  # 4th highest value; idx. 25
                    9.38450987,  # 1st highest value; idx. 26
                    2.12662941,
                    -9.32562038,
                    2.35652522,
                ],  # cummulative prob of 5 highest values <= 0.6
                [
                    0.58425518,
                    4.53139238,
                    -5.57510464,
                    -6.28030699,
                    -7.19529503,
                    -4.02122551,
                    1.39337037,
                    -6.06707057,
                    1.59480517,
                    -9.643119,
                    0.03907799,
                    0.67231762,
                    -8.88206726,
                    6.27115922,  # 4th highest value; idx. 13
                    2.28520723,
                    4.82767506,
                    4.30421368,
                    8.8275313,  # 2nd highest value; idx. 17
                    5.44029958,  # 5th highest value; idx. 18
                    -4.4735794,
                    7.38579536,  # 3rd highest value; idx. 20
                    -2.91051663,
                    2.61946077,
                    -2.5674762,
                    -9.48959302,
                    -4.02922645,
                    -1.35416918,
                    9.67702323,  # 1st highest value; idx. 27
                    -5.89478553,
                    1.85370467,
                ],  # cummulative prob of 5 highest values <= 0.6
            ],
            dtype=tf.float32,
        )

        non_inf_expected_idx = tf.convert_to_tensor(
            [[0, 0], [0, 9], [0, 10], [0, 25], [0, 26], [1, 13], [1, 17], [1, 18], [1, 20], [1, 27]], dtype=tf.int32,
        )  # expected non filtered idx as noted above

        non_inf_expected_output = tf.convert_to_tensor(
            [8.222099, 7.3534126, 8.432078, 7.4402075, 9.38451, 6.271159, 8.827531, 5.4402995, 7.3857956, 9.677023],
            dtype=tf.float32,
        )  # expected non filtered values as noted above

        output = sampler.warp_tf([], logits)

        non_inf_output = output[output != -float("inf")]
        non_inf_idx = tf.cast(
            tf.where(tf.not_equal(output, tf.constant(-float("inf"), dtype=tf.float32))), dtype=tf.int32,
        )

        tf.debugging.assert_near(non_inf_output, non_inf_expected_output, rtol=1e-12)
        tf.debugging.assert_equal(non_inf_idx, non_inf_expected_idx)
