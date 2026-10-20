# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from parameterized import parameterized

from transformers import (
    AutoformerForPrediction,
    InformerForPrediction,
    TimeSeriesTransformerForPrediction,
    is_torch_available,
)
from transformers.testing_utils import require_torch, torch_device

from ..test_modeling_common import floats_tensor


if is_torch_available():
    import torch


@require_torch
class TimeSeriesGenerationTest(unittest.TestCase):
    @parameterized.expand(
        [
            (model_class, distribution_output, input_size)
            for model_class in (
                TimeSeriesTransformerForPrediction,
                InformerForPrediction,
                AutoformerForPrediction,
            )
            for distribution_output in ("student_t", "normal", "negative_binomial")
            for input_size in (1, 2)
        ]
    )
    def test_generate_distribution_params(self, model_class, distribution_output, input_size):
        config = model_class.config_class(
            prediction_length=3,
            context_length=12,
            lags_sequence=[1],
            input_size=input_size,
            num_time_features=1,
            num_parallel_samples=2,
            encoder_layers=1,
            decoder_layers=1,
            distribution_output=distribution_output,
            scaling="mean" if distribution_output == "negative_binomial" else "std",
        )
        model = model_class(config).to(torch_device).eval()
        target_shape = [] if input_size == 1 else [input_size]
        past_values = floats_tensor([2, model.model._past_length] + target_shape) + 10
        inputs = {
            "past_values": past_values,
            "past_observed_mask": torch.ones_like(past_values),
            "past_time_features": floats_tensor([2, model.model._past_length, 1]),
            "future_time_features": floats_tensor([2, config.prediction_length, 1]),
        }
        projected_params = []

        def capture_params(module, args, output):
            projected_params.append(tuple(param.clone() for param in output))

        with model.parameter_projection.register_forward_hook(capture_params):
            outputs = model.generate(**inputs)

        self.assertEqual(len(outputs.params), 3 if distribution_output == "student_t" else 2)
        self.assertEqual(
            outputs.sequences.shape, (2, config.num_parallel_samples, config.prediction_length, *target_shape)
        )
        for param, steps in zip(outputs.params, zip(*projected_params)):
            self.assertEqual(param.shape, outputs.sequences.shape)
            torch.testing.assert_close(param, torch.cat(steps, dim=1).reshape(param.shape))

        _, loc, scale = model.model.scaler(
            past_values[:, -config.context_length :], inputs["past_observed_mask"][:, -config.context_length :]
        )
        torch.testing.assert_close(outputs.loc, loc.unsqueeze(1))
        torch.testing.assert_close(outputs.scale, scale.unsqueeze(1))
        distribution = model.output_distribution(outputs.params, loc=outputs.loc, scale=outputs.scale)
        self.assertEqual(distribution.sample().shape, outputs.sequences.shape)
        self.assertTrue(torch.isfinite(distribution.log_prob(outputs.sequences)).all())
