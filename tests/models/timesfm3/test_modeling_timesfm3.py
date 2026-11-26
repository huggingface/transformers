# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import math
import unittest

import torch
from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import TimesFm3Config, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION, ModelTesterMixin


if is_torch_available():
    from transformers import TimesFm3Model, TimesFm3ModelForPrediction


class TimesFm3ModelTester:
    def __init__(
        self,
        parent,
        patch_length: int = 8,
        horizon_length: int = 16,
        context_length: int = 64,
        hidden_size: int = 32,
        intermediate_size: int = 48,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
        batch_size: int = 2,
        num_targets: int = 2,
        sequence_length: int = 40,
        prediction_length: int = 12,
        is_training: bool = False,
    ):
        self.parent = parent
        self.patch_length = patch_length
        self.horizon_length = horizon_length
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.quantiles = list(quantiles)
        self.batch_size = batch_size
        self.num_targets = num_targets
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.is_training = is_training

        # The model reads the patches of the context followed by those of the horizon. Without known covariates,
        # the horizon is rounded up to a multiple of `horizon_length`: here, 16 steps predicted from the last
        # patch of the context, which reads one more patch after the one it starts at.
        self.seq_length = sequence_length // patch_length + horizon_length // patch_length

    def get_config(self, **kwargs):
        config_kwargs = {
            "patch_length": self.patch_length,
            "horizon_length": self.horizon_length,
            "context_length": self.context_length,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_attention_heads,
            "head_dim": self.hidden_size // self.num_attention_heads,
            "quantiles": self.quantiles,
            "decode_index": len(self.quantiles) // 2,
        }
        config_kwargs.update(kwargs)
        return TimesFm3Config(**config_kwargs)

    def get_pipeline_config(self):
        return self.get_config()

    def prepare_series(self, num_variates: int, length: int, seed: int = 0, positive: bool = False):
        generator = torch.Generator().manual_seed(seed)
        steps = torch.arange(length, dtype=torch.float32)
        noise = torch.randn(self.batch_size, num_variates, length, generator=generator).cumsum(-1)
        series = 10 * torch.sin(2 * math.pi * steps / 24) + noise
        if positive:
            series = series.abs() + 1
        return series.to(torch_device)

    def prepare_config_and_inputs(self):
        return self.get_config(), self.prepare_series(self.num_targets, self.sequence_length)

    def prepare_config_and_inputs_for_common(self):
        config, past_values = self.prepare_config_and_inputs()
        inputs_dict = {"past_values": past_values, "prediction_length": self.prediction_length}
        return config, inputs_dict


@require_torch
class TimesFm3ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (TimesFm3ModelForPrediction,) if is_torch_available() else ()
    test_resize_embeddings = False
    is_encoder_decoder = False
    test_inputs_embeds = False

    def setUp(self):
        self.model_tester = TimesFm3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TimesFm3Config, has_text_modality=False)

    def _get_model(self, **config_kwargs):
        config = self.model_tester.get_config(**config_kwargs)
        return TimesFm3ModelForPrediction(config).to(torch_device).eval()

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_config_validation(self):
        # Each patch has to predict more steps than it holds, for the predictions of consecutive patches to
        # overlap.
        with self.assertRaises(StrictDataclassClassValidationError):
            TimesFm3Config(patch_length=32, horizon_length=32)
        with self.assertRaises(StrictDataclassClassValidationError):
            TimesFm3Config(patch_length=32, horizon_length=48)
        with self.assertRaises(StrictDataclassClassValidationError):
            TimesFm3Config(quantiles=[0.1, 0.5, 0.9], decode_index=4)

    def test_create_and_run_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = TimesFm3ModelForPrediction(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**inputs_dict)

        tester = self.model_tester
        expected_shape = (tester.batch_size, tester.num_targets, tester.prediction_length, len(tester.quantiles))
        self.assertEqual(outputs.full_predictions.shape, expected_shape)
        self.assertEqual(outputs.mean_predictions.shape, expected_shape[:-1])
        self.assertEqual(
            outputs.last_hidden_state.shape,
            (tester.batch_size, tester.num_targets, tester.seq_length, tester.hidden_size),
        )
        self.assertTrue(torch.isfinite(outputs.full_predictions).all())
        self.assertTrue((outputs.full_predictions.diff(dim=-1) >= 0).all(), "quantiles must be sorted")
        torch.testing.assert_close(outputs.mean_predictions, outputs.full_predictions[..., config.decode_index])

    def test_base_model(self):
        config, past_values = self.model_tester.prepare_config_and_inputs()
        tester = self.model_tester
        model = TimesFm3Model(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(past_values=past_values)

        num_patches = tester.sequence_length // tester.patch_length
        self.assertEqual(
            outputs.last_hidden_state.shape, (tester.batch_size, tester.num_targets, num_patches, tester.hidden_size)
        )
        for statistics in (outputs.context_count, outputs.context_mu, outputs.context_sigma):
            self.assertEqual(statistics.shape, (tester.batch_size, tester.num_targets, num_patches))
        # The statistics of a patch cover the series up to the end of the patch.
        self.assertTrue((outputs.context_count[..., -1] == tester.sequence_length).all())
        torch.testing.assert_close(outputs.context_mu[..., -1], past_values.mean(-1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            outputs.context_sigma[..., -1], past_values.std(-1, correction=0), rtol=1e-4, atol=1e-4
        )

        with self.assertRaises(ValueError):
            model(past_values=past_values[..., :-1])

    def test_base_model_is_causal(self):
        config, past_values = self.model_tester.prepare_config_and_inputs()
        model = TimesFm3Model(config).to(torch_device).eval()
        changed = past_values.clone()
        changed[..., -self.model_tester.patch_length :] += 1.0
        with torch.no_grad():
            first = model(past_values=past_values).last_hidden_state
            second = model(past_values=changed).last_hidden_state
        # Changing the last patch leaves the hidden states of the patches before it unchanged.
        torch.testing.assert_close(first[:, :, :-1], second[:, :, :-1])
        self.assertFalse(torch.allclose(first[:, :, -1], second[:, :, -1]))

    @parameterized.expand(
        [
            ("one_patch", 1, 40, 12),
            ("context_past_the_cap", 3, 100, 20),
            ("several_patches_after_a_short_context", 2, 13, 70),
            ("single_step", 1, 24, 1),
            ("context_shorter_than_a_patch", 1, 3, 5),
        ]
    )
    def test_prediction_shapes(self, _, num_targets, sequence_length, prediction_length):
        model = self._get_model()
        past_values = self.model_tester.prepare_series(num_targets, sequence_length)
        past_values[0, 0, 1:2] = float("nan")
        with torch.no_grad():
            outputs = model(past_values=past_values, prediction_length=prediction_length)

        num_quantiles = len(self.model_tester.quantiles)
        self.assertEqual(
            outputs.full_predictions.shape,
            (self.model_tester.batch_size, num_targets, prediction_length, num_quantiles),
        )
        self.assertTrue(torch.isfinite(outputs.full_predictions).all())
        self.assertTrue((outputs.full_predictions.diff(dim=-1) >= 0).all(), "quantiles must be sorted")

    def test_input_shapes(self):
        model = self._get_model()
        past_values = self.model_tester.prepare_series(1, 40)
        with torch.no_grad():
            expected = model(past_values=past_values, prediction_length=12).full_predictions
            two_dimensional = model(past_values=past_values[:, 0], prediction_length=12).full_predictions
            one_dimensional = model(past_values=past_values[0, 0], prediction_length=12).full_predictions
        torch.testing.assert_close(two_dimensional, expected)
        torch.testing.assert_close(one_dimensional, expected[:1], rtol=1e-4, atol=1e-4)

        with self.assertRaises(ValueError):
            model(past_values=past_values[None], prediction_length=12)
        with self.assertRaises(ValueError):
            model(past_values=past_values, prediction_length=0)

    def test_default_prediction_length(self):
        model = self._get_model()
        past_values = self.model_tester.prepare_series(1, 40)
        with torch.no_grad():
            outputs = model(past_values=past_values)
        self.assertEqual(outputs.mean_predictions.shape[-1], self.model_tester.horizon_length)

    def test_longer_forecast_starts_with_shorter_one(self):
        # Both forecasts are read from the same pass over the patches, whose attention over time is causal.
        model = self._get_model()
        past_values = self.model_tester.prepare_series(2, 40)
        with torch.no_grad():
            short = model(past_values=past_values, prediction_length=10).full_predictions
            long = model(past_values=past_values, prediction_length=16).full_predictions
        torch.testing.assert_close(long[:, :, :10], short, rtol=1e-4, atol=1e-4)

    def test_missing_values(self):
        model = self._get_model()
        tester = self.model_tester
        past_values = tester.prepare_series(2, 40)

        # Missing values inside a series are interpolated linearly.
        with_gap = past_values.clone()
        with_gap[0, 0, 10:13] = float("nan")
        filled = past_values.clone()
        filled[0, 0, 9:14] = torch.linspace(past_values[0, 0, 9], past_values[0, 0, 13], 5)
        # The time steps before the first observation of a batch element are ignored, whatever they hold.
        with_start = past_values.clone()
        with_start[1, :, :8] = float("nan")
        shifted = past_values.clone()
        shifted[1, :, :8] += 100.0
        with torch.no_grad():
            from_gap = model(past_values=with_gap, prediction_length=12).full_predictions
            from_filled = model(past_values=filled, prediction_length=12).full_predictions
            from_start = model(past_values=with_start, prediction_length=12).full_predictions
            from_shifted = model(past_values=shifted, prediction_length=12).full_predictions
            reference = model(past_values=past_values, prediction_length=12).full_predictions
        torch.testing.assert_close(from_gap, from_filled, rtol=1e-4, atol=1e-4)
        self.assertTrue(torch.isfinite(from_start).all())
        torch.testing.assert_close(from_start[0], reference[0], rtol=1e-4, atol=1e-4)
        self.assertFalse(torch.allclose(from_start[1], reference[1]))
        self.assertFalse(torch.allclose(from_start[1], from_shifted[1]))

        # A series without any observation is forecast as a series of zeros.
        empty = torch.full_like(past_values, float("nan"))
        with torch.no_grad():
            from_empty = model(past_values=empty, prediction_length=12).full_predictions
            from_zeros = model(past_values=torch.zeros_like(past_values), prediction_length=12).full_predictions
        torch.testing.assert_close(from_empty, from_zeros)

    def test_leading_missing_values_in_the_whole_batch_are_dropped(self):
        model = self._get_model()
        past_values = self.model_tester.prepare_series(2, 40)
        with_start = past_values.clone()
        with_start[:, :, :11] = float("nan")
        with torch.no_grad():
            from_start = model(past_values=with_start, prediction_length=12).full_predictions
            from_shorter = model(past_values=past_values[..., 11:], prediction_length=12).full_predictions
        torch.testing.assert_close(from_start, from_shorter)

    def test_series_of_a_batch_are_independent(self):
        model = self._get_model()
        past_values = self.model_tester.prepare_series(2, 40)
        with torch.no_grad():
            batched = model(past_values=past_values, prediction_length=20).full_predictions
            single = model(past_values=past_values[1:], prediction_length=20).full_predictions
        torch.testing.assert_close(batched[1:], single, rtol=1e-4, atol=1e-4)

    def test_variates_of_a_series_interact(self):
        model = self._get_model()
        past_values = self.model_tester.prepare_series(2, 40)
        other = past_values.clone()
        other[:, 1] = self.model_tester.prepare_series(1, 40, seed=1)[:, 0]
        with torch.no_grad():
            first = model(past_values=past_values, prediction_length=12).full_predictions
            second = model(past_values=other, prediction_length=12).full_predictions
        self.assertFalse(torch.allclose(first[:, 0], second[:, 0]))

    def test_covariates(self):
        model = self._get_model()
        tester = self.model_tester
        past_values = tester.prepare_series(1, 40, seed=3)
        past_covariates = tester.prepare_series(2, 40, seed=1)
        known_covariates = tester.prepare_series(1, 40 + 20, seed=2)
        past_covariates[0, 0, 10:14] = float("nan")
        with torch.no_grad():
            outputs = model(
                past_values=past_values,
                prediction_length=20,
                past_covariates=past_covariates,
                known_covariates=known_covariates,
            )
            without_covariates = model(past_values=past_values, prediction_length=20)

        # Only the targets are forecast.
        self.assertEqual(outputs.full_predictions.shape, (tester.batch_size, 1, 20, len(tester.quantiles)))
        self.assertEqual(outputs.last_hidden_state.shape[1], 4)
        self.assertTrue(torch.isfinite(outputs.full_predictions).all())
        self.assertTrue((outputs.full_predictions.diff(dim=-1) >= 0).all(), "quantiles must be sorted")
        self.assertFalse(torch.allclose(outputs.full_predictions, without_covariates.full_predictions))

        with self.assertRaises(ValueError):
            model(past_values=past_values, prediction_length=20, known_covariates=known_covariates[..., :-1])
        with self.assertRaises(ValueError):
            model(past_values=past_values, prediction_length=20, past_covariates=known_covariates)

    def test_only_known_covariates_reach_targets_from_the_future(self):
        model = self._get_model()
        tester = self.model_tester
        past_values = tester.prepare_series(1, 40, seed=3)
        known_covariates = tester.prepare_series(1, 40 + 12, seed=2)
        changed = known_covariates.clone()
        changed[..., -4:] += 5.0
        with torch.no_grad():
            first = model(past_values=past_values, prediction_length=12, known_covariates=known_covariates)
            second = model(past_values=past_values, prediction_length=12, known_covariates=changed)
            # The same series, when they are only known over the context, are cut at the end of the context.
            past_first = model(
                past_values=past_values, prediction_length=12, past_covariates=known_covariates[..., :40]
            )
            past_second = model(past_values=past_values, prediction_length=12, past_covariates=changed[..., :40])
        self.assertFalse(torch.allclose(first.full_predictions, second.full_predictions))
        torch.testing.assert_close(past_first.full_predictions, past_second.full_predictions)

    def test_linear_trend_is_extrapolated(self):
        # The trend of a series it explains well is removed before the model and added back to the forecast:
        # adding a steep line to such a series shifts its forecast by the continuation of the line.
        model = self._get_model()
        steps = torch.arange(40 + 16, dtype=torch.float32, device=torch_device)
        slope = 3.0
        past_values = (slope * steps[:40] + torch.sin(steps[:40]))[None, None]
        with torch.no_grad():
            first = model(past_values=past_values, prediction_length=16).mean_predictions
            second = model(past_values=past_values + 50.0 * steps[:40], prediction_length=16).mean_predictions
        expected_shift = 50.0 * steps[40:]
        torch.testing.assert_close(second - first, expected_shift[None, None], rtol=1e-2, atol=1.0)

    def test_flip_invariance(self):
        # The second pass negates the covariates too, so negating every input reflects the forecast.
        model = self._get_model()
        tester = self.model_tester
        past_values = tester.prepare_series(1, 40, seed=3)
        past_covariates = tester.prepare_series(2, 40, seed=1)
        known_covariates = tester.prepare_series(1, 40 + 20, seed=2)
        with torch.no_grad():
            positive = model(
                past_values=past_values,
                prediction_length=20,
                past_covariates=past_covariates,
                known_covariates=known_covariates,
                force_flip_invariance=True,
            ).full_predictions
            negative = model(
                past_values=-past_values,
                prediction_length=20,
                past_covariates=-past_covariates,
                known_covariates=-known_covariates,
                force_flip_invariance=True,
            ).full_predictions
            single_pass = model(
                past_values=past_values,
                prediction_length=20,
                past_covariates=past_covariates,
                known_covariates=known_covariates,
            ).full_predictions
        torch.testing.assert_close(negative, -positive.flip(-1), rtol=0, atol=0)
        self.assertFalse(torch.allclose(positive, single_pass))

    def test_truncate_negative(self):
        model = self._get_model()
        past_values = self.model_tester.prepare_series(1, 40, positive=True)
        # A strong downward trend pushes the unclipped forecast of the first series below 0.
        past_values[0, 0] = torch.linspace(50, 0.5, 40, device=torch_device)
        with torch.no_grad():
            clipped = model(past_values=past_values, prediction_length=40, truncate_negative=True).full_predictions
            unclipped = model(past_values=past_values, prediction_length=40).full_predictions
            # A series with a negative value is never clipped.
            mixed = past_values.clone()
            mixed[1, 0, 0] = -1.0
            not_clipped = model(past_values=mixed, prediction_length=40, truncate_negative=True).full_predictions
            reference = model(past_values=mixed, prediction_length=40).full_predictions
        self.assertTrue((unclipped[0] < 0).any())
        self.assertTrue((clipped >= 0).all())
        torch.testing.assert_close(clipped, unclipped.clamp_min(0))
        torch.testing.assert_close(not_clipped[1], reference[1])

    def test_config_defaults_of_the_forecast_options(self):
        tester = self.model_tester
        past_values = tester.prepare_series(1, 40, positive=True)
        past_values[0, 0] = torch.linspace(50, 0.5, 40, device=torch_device)
        model = self._get_model()
        from_config = self._get_model(force_flip_invariance=True, infer_is_positive=True)
        from_config.load_state_dict(model.state_dict())
        with torch.no_grad():
            expected = model(
                past_values=past_values, prediction_length=40, force_flip_invariance=True, truncate_negative=True
            ).full_predictions
            predictions = from_config(past_values=past_values, prediction_length=40).full_predictions
        torch.testing.assert_close(predictions, expected)

    @unittest.skip(reason="Model does not have input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    def test_hidden_states_output(self):
        """
        The hidden states keep the grid of the patches of every series: they have shape `(batch_size,
        num_variates, num_patches, hidden_size)`, where the generic test expects a single sequence dimension.
        """
        tester = self.model_tester
        config, inputs_dict = tester.prepare_config_and_inputs_for_common()
        expected_shape = (tester.batch_size, tester.num_targets, tester.seq_length, tester.hidden_size)

        for from_config in (False, True):
            config.output_hidden_states = from_config
            model = TimesFm3ModelForPrediction(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict) if from_config else model(**inputs_dict, output_hidden_states=True)

            # The output of the patch embedding, then the output of each layer.
            self.assertEqual(len(outputs.hidden_states), tester.num_hidden_layers + 1)
            for hidden_states in outputs.hidden_states:
                self.assertEqual(hidden_states.shape, expected_shape)
            torch.testing.assert_close(outputs.hidden_states[-1], outputs.last_hidden_state)

    def test_attention_outputs(self):
        """
        A layer runs an attention over time and an attention over the variates, so the model records two
        attention maps per layer, with a shape that depends on the axis.
        """
        tester = self.model_tester
        config, inputs_dict = tester.prepare_config_and_inputs_for_common()
        time_shape = (
            tester.batch_size * tester.num_targets,
            tester.num_attention_heads,
            tester.seq_length,
            tester.seq_length,
        )
        variate_shape = (
            tester.batch_size * tester.seq_length,
            tester.num_attention_heads,
            tester.num_targets,
            tester.num_targets,
        )

        for from_config in (False, True):
            config.output_attentions = from_config
            model = TimesFm3ModelForPrediction._from_config(config, attn_implementation="eager")
            model.to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict) if from_config else model(**inputs_dict, output_attentions=True)

            self.assertEqual(len(outputs.attentions), 2 * tester.num_hidden_layers)
            for index, attention in enumerate(outputs.attentions):
                self.assertEqual(attention.shape, time_shape if index % 2 == 0 else variate_shape)
                # Every patch can at least attend itself.
                torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention.sum(-1)))
            # The attention over time is causal.
            self.assertTrue((outputs.attentions[0].triu(diagonal=1) == 0).all())

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        """
        TimesFM 3 builds its attention masks from the missing values of the series, so the generic test, which
        injects an attention mask, does not apply. This override checks the equivalence of eager and SDPA directly.
        """
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        # The two implementations round the attention weights differently, the more so with fewer mantissa bits.
        tolerance = {torch.float32: 1e-5, torch.float16: 2e-2, torch.bfloat16: 1e-1}[dtype]

        # One config per model: the attention implementation is stored on the config.
        model_eager = TimesFm3ModelForPrediction._from_config(
            self.model_tester.get_config(), attn_implementation="eager"
        )
        model_eager.to(dtype=dtype, device=torch_device).eval()
        model_sdpa = TimesFm3ModelForPrediction._from_config(
            self.model_tester.get_config(), attn_implementation="sdpa"
        )
        model_sdpa.load_state_dict(model_eager.state_dict())
        model_sdpa.to(dtype=dtype, device=torch_device).eval()
        self.assertEqual(model_eager.config._attn_implementation, "eager")
        self.assertEqual(model_sdpa.config._attn_implementation, "sdpa")

        def relative_difference(first: torch.Tensor, second: torch.Tensor) -> float:
            first, second = first.float(), second.float()
            return ((first - second).norm() / second.norm()).item()

        # Leading missing values in one series only hide its first patches from the attention.
        past_values = self.model_tester.prepare_series(2, 40)
        past_values[0, :, :19] = float("nan")
        known_covariates = self.model_tester.prepare_series(1, 40 + 12, seed=2)
        for covariates in ({}, {"known_covariates": known_covariates}):
            with torch.no_grad():
                outputs_eager = model_eager(
                    past_values=past_values, prediction_length=12, output_hidden_states=True, **covariates
                )
                outputs_sdpa = model_sdpa(
                    past_values=past_values, prediction_length=12, output_hidden_states=True, **covariates
                )
            # The hidden patches themselves are not compared: nothing reads them.
            hidden_states_eager = outputs_eager.hidden_states[-1][:, :, 2:]
            hidden_states_sdpa = outputs_sdpa.hidden_states[-1][:, :, 2:]
            self.assertLess(relative_difference(hidden_states_eager, hidden_states_sdpa), tolerance)
            self.assertLess(
                relative_difference(outputs_eager.full_predictions, outputs_sdpa.full_predictions), tolerance
            )


@require_torch
@slow
class TimesFm3ModelIntegrationTests(unittest.TestCase):
    @unittest.skip(
        reason="The converted checkpoint is not on the Hub yet. The expected values come from "
        "`google/timesfm-3.0-pytorch` converted with `convert_timesfm3_original_to_hf.py`."
    )
    def test_inference(self):
        model = TimesFm3ModelForPrediction.from_pretrained("google/timesfm-3.0-pytorch").to(torch_device)
        steps = torch.arange(512, dtype=torch.float32, device=torch_device)
        past_values = torch.sin(2 * math.pi * steps / 24)[None, None, :]

        with torch.no_grad():
            outputs = model(past_values=past_values, prediction_length=64)

        self.assertEqual(outputs.mean_predictions.shape, (1, 1, 64))
        self.assertEqual(outputs.full_predictions.shape, (1, 1, 64, 9))
        self.assertEqual((model.config.quantiles[0], model.config.quantiles[8]), (0.1, 0.9))

        # The model carries on the sine wave.
        expected_median_start = torch.tensor(
            [0.864650, 0.703734, 0.498623, 0.258608, 0.001887, -0.256365, -0.496276, -0.704066]
        )
        expected_median_end = torch.tensor([-0.865357, -0.708215, -0.503280, -0.262230])
        expected_quantile_10 = torch.tensor([0.861019, 0.700785, 0.495857, 0.252802])
        expected_quantile_90 = torch.tensor([0.868762, 0.708234, 0.501612, 0.260958])

        median = outputs.mean_predictions[0, 0].cpu()
        quantiles = outputs.full_predictions[0, 0].cpu()
        torch.testing.assert_close(median[:8], expected_median_start, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(median[-4:], expected_median_end, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(quantiles[:4, 0], expected_quantile_10, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(quantiles[:4, 8], expected_quantile_90, rtol=1e-4, atol=1e-4)
