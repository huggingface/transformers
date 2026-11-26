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

from transformers import TimestralConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION, ModelTesterMixin


if is_torch_available():
    from transformers import TimestralModel, TimestralModelForPrediction


class TimestralModelTester:
    def __init__(
        self,
        parent,
        patch_length: int = 8,
        context_length: int = 64,
        block_size: int = 2,
        hidden_size: int = 32,
        num_attention_heads: int = 4,
        intermediate_size: int = 48,
        block_types: tuple[str, ...] = ("time", "time_variate"),
        quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
        batch_size: int = 2,
        num_targets: int = 2,
        sequence_length: int = 40,
        prediction_length: int = 12,
        is_training: bool = False,
    ):
        self.parent = parent
        self.patch_length = patch_length
        self.context_length = context_length
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.block_types = list(block_types)
        self.num_hidden_layers = len(block_types)
        self.quantiles = list(quantiles)
        self.batch_size = batch_size
        self.num_targets = num_targets
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.is_training = is_training

        # One block: the context patches followed by one forecast token per predicted patch.
        self.num_prediction_patches = math.ceil(prediction_length / patch_length)
        self.seq_length = sequence_length // patch_length + self.num_prediction_patches

    def get_config(self, **kwargs):
        config_kwargs = {
            "patch_length": self.patch_length,
            "context_length": self.context_length,
            "block_size": self.block_size,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "block_types": self.block_types,
            "quantiles": self.quantiles,
        }
        config_kwargs.update(kwargs)
        return TimestralConfig(**config_kwargs)

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
class TimestralModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (TimestralModelForPrediction,) if is_torch_available() else ()
    test_resize_embeddings = False
    is_encoder_decoder = False
    test_inputs_embeds = False

    def setUp(self):
        self.model_tester = TimestralModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TimestralConfig, has_text_modality=False)

    def _get_model(self, **config_kwargs):
        config = self.model_tester.get_config(**config_kwargs)
        return TimestralModelForPrediction(config).to(torch_device).eval()

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_config_default_block_types(self):
        config = TimestralConfig(num_hidden_layers=8)
        self.assertEqual(config.block_types, ["time", "time", "time", "time_variate"] * 2)
        config = TimestralConfig(block_types=["time", "variate"])
        self.assertEqual(config.num_hidden_layers, 2)
        self.assertEqual(config.block_axes, [(True,), (False,)])

    def test_config_known_covariates_support(self):
        self.assertTrue(TimestralConfig(block_types=["time", "time_variate"]).supports_known_covariates)
        # The future of a known covariate cannot reach a target without an attention over time and one over
        # variates after an attention over time.
        self.assertFalse(TimestralConfig(block_types=["time", "time"]).supports_known_covariates)
        self.assertFalse(TimestralConfig(block_types=["variate", "time"]).supports_known_covariates)
        self.assertFalse(
            TimestralConfig(block_types=["time", "time_variate"], use_covariates=False).supports_known_covariates
        )

    def test_config_validation(self):
        with self.assertRaises(StrictDataclassClassValidationError):
            TimestralConfig(block_types=["time", "space"])
        with self.assertRaises(StrictDataclassClassValidationError):
            TimestralConfig(quantiles=[0.1, 0.9])
        with self.assertRaises(StrictDataclassClassValidationError):
            TimestralConfig(quantiles=[0.5, 0.1, 0.9])
        with self.assertRaises(StrictDataclassClassValidationError):
            TimestralConfig(hidden_size=30, num_attention_heads=4)

    def test_create_and_run_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = TimestralModelForPrediction(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**inputs_dict)

        tester = self.model_tester
        expected_shape = (tester.batch_size, tester.num_targets, tester.prediction_length, len(tester.quantiles))
        self.assertEqual(outputs.full_predictions.shape, expected_shape)
        self.assertEqual(outputs.mean_predictions.shape, expected_shape[:-1])
        self.assertTrue(torch.isfinite(outputs.full_predictions).all())
        self.assertTrue((outputs.full_predictions.diff(dim=-1) >= 0).all(), "quantiles must be sorted")
        torch.testing.assert_close(outputs.mean_predictions, outputs.full_predictions[..., config.median_index])

    def test_base_model(self):
        config, past_values = self.model_tester.prepare_config_and_inputs()
        model = TimestralModel(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(past_values=past_values, num_prediction_patches=2)
        tester = self.model_tester
        self.assertEqual(
            outputs.last_hidden_state.shape, (tester.batch_size, tester.num_targets, 2, tester.hidden_size)
        )
        self.assertEqual(outputs.loc.shape, (tester.batch_size, tester.num_targets))
        self.assertEqual(outputs.scale.shape, (tester.batch_size, tester.num_targets))

        with self.assertRaises(ValueError):
            model(past_values=past_values[..., :-1])

    @parameterized.expand(
        [
            ("one_block", 1, 40, 12),
            ("context_past_the_cap", 3, 100, 20),
            ("three_blocks_after_a_short_context", 2, 13, 40),
            ("single_step", 1, 24, 1),
        ]
    )
    def test_prediction_shapes(self, _, num_targets, sequence_length, prediction_length):
        model = self._get_model()
        past_values = self.model_tester.prepare_series(num_targets, sequence_length)
        past_values[0, 0, 5:9] = float("nan")
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

    def test_default_prediction_length(self):
        model = self._get_model()
        past_values = self.model_tester.prepare_series(1, 40)
        with torch.no_grad():
            outputs = model(past_values=past_values)
        one_block = self.model_tester.block_size * self.model_tester.patch_length
        self.assertEqual(outputs.mean_predictions.shape[-1], one_block)

    def test_missing_values(self):
        model = self._get_model()
        past_values = self.model_tester.prepare_series(2, 40)
        observed_mask = torch.ones_like(past_values, dtype=torch.bool)
        observed_mask[0, 0, 5:9] = False
        observed_mask[1, 1, :3] = False
        with_nan = past_values.masked_fill(~observed_mask, float("nan"))
        with torch.no_grad():
            from_nan = model(past_values=with_nan, prediction_length=12).full_predictions
            from_mask = model(
                past_values=past_values, past_observed_mask=observed_mask, prediction_length=12
            ).full_predictions
        torch.testing.assert_close(from_nan, from_mask)
        self.assertTrue(torch.isfinite(from_nan).all())

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
        self.assertTrue(torch.isfinite(outputs.full_predictions).all())
        self.assertTrue((outputs.full_predictions.diff(dim=-1) >= 0).all(), "quantiles must be sorted")
        self.assertFalse(torch.allclose(outputs.full_predictions, without_covariates.full_predictions))

        with self.assertRaises(ValueError):
            model(past_values=past_values, prediction_length=20, known_covariates=known_covariates[..., :-1])

    def test_known_covariates_future_reaches_targets(self):
        model = self._get_model()
        tester = self.model_tester
        past_values = tester.prepare_series(1, 40, seed=3)
        known_covariates = tester.prepare_series(1, 40 + 12, seed=2)
        changed = known_covariates.clone()
        changed[..., -4:] += 5.0
        with torch.no_grad():
            first = model(past_values=past_values, prediction_length=12, known_covariates=known_covariates)
            second = model(past_values=past_values, prediction_length=12, known_covariates=changed)
        # Changing only the last steps of the horizon of a known covariate changes the forecast of the target.
        self.assertFalse(torch.allclose(first.full_predictions, second.full_predictions))

    def test_known_covariates_need_a_supporting_schedule(self):
        model = self._get_model(block_types=["time", "time"])
        past_values = self.model_tester.prepare_series(1, 40)
        known_covariates = self.model_tester.prepare_series(1, 40 + 12, seed=2)
        with self.assertRaises(ValueError):
            model(past_values=past_values, prediction_length=12, known_covariates=known_covariates)

        model = self._get_model(use_covariates=False)
        with self.assertRaises(ValueError):
            model(past_values=past_values, prediction_length=12, past_covariates=past_values)

    def test_symmetric_averaging_is_odd(self):
        # The mirrored pass negates the covariates too, so negating every input reflects the forecast.
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
                symmetric_averaging=True,
                clip_nonnegative=False,
            ).full_predictions
            negative = model(
                past_values=-past_values,
                prediction_length=20,
                past_covariates=-past_covariates,
                known_covariates=-known_covariates,
                symmetric_averaging=True,
                clip_nonnegative=False,
            ).full_predictions
        torch.testing.assert_close(negative, -positive.flip(-1), rtol=0, atol=0)

    def test_clip_nonnegative(self):
        model = self._get_model()
        past_values = self.model_tester.prepare_series(1, 40, positive=True)
        # A strong downward trend pushes the unclipped forecast of the first series below 0.
        past_values[0, 0] = torch.linspace(50, 0.5, 40, device=torch_device)
        with torch.no_grad():
            clipped = model(past_values=past_values, prediction_length=40, clip_nonnegative=True).full_predictions
            unclipped = model(past_values=past_values, prediction_length=40, clip_nonnegative=False).full_predictions
            # A series with a negative value is never clipped.
            mixed = past_values.clone()
            mixed[1, 0, 0] = -1.0
            not_clipped = model(past_values=mixed, prediction_length=40, clip_nonnegative=True).full_predictions
            reference = model(past_values=mixed, prediction_length=40, clip_nonnegative=False).full_predictions
        self.assertTrue((clipped >= 0).all())
        torch.testing.assert_close(clipped, unclipped.clamp_min(0))
        torch.testing.assert_close(not_clipped[1], reference[1])

    @unittest.skip(reason="Model does not have input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    def test_attention_outputs(self):
        """
        A layer runs one attention per axis of its block type, so the model records one attention map per
        attention sublayer rather than one per layer, with a shape that depends on the axis.
        """
        tester = self.model_tester
        config, inputs_dict = tester.prepare_config_and_inputs_for_common()
        axes = [is_time for layer_axes in config.block_axes for is_time in layer_axes]
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
            model = TimestralModelForPrediction._from_config(config, attn_implementation="eager")
            model.to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict) if from_config else model(**inputs_dict, output_attentions=True)

            self.assertEqual(len(outputs.attentions), len(axes))
            for attention, is_time in zip(outputs.attentions, axes):
                self.assertEqual(attention.shape, time_shape if is_time else variate_shape)
                # Every token can at least attend itself.
                torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention.sum(-1)))

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        """
        Timestral builds its attention masks from the observation mask of the series, so the generic test, which
        injects an attention mask, does not apply. This override checks the equivalence of eager and SDPA directly.
        """
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        tolerance = {torch.float32: 1e-4, torch.bfloat16: 5e-2, torch.float16: 5e-2}[dtype]

        # One config per model: the attention implementation is stored on the config.
        model_eager = TimestralModelForPrediction._from_config(
            self.model_tester.get_config(), attn_implementation="eager"
        )
        model_eager.to(dtype=dtype, device=torch_device).eval()
        model_sdpa = TimestralModelForPrediction._from_config(
            self.model_tester.get_config(), attn_implementation="sdpa"
        )
        model_sdpa.load_state_dict(model_eager.state_dict())
        model_sdpa.to(dtype=dtype, device=torch_device).eval()
        self.assertEqual(model_eager.config._attn_implementation, "eager")
        self.assertEqual(model_sdpa.config._attn_implementation, "sdpa")

        past_values = self.model_tester.prepare_series(2, 40)
        past_values[0, 0, 5:9] = float("nan")
        known_covariates = self.model_tester.prepare_series(1, 40 + 12, seed=2)
        for covariates in ({}, {"known_covariates": known_covariates}):
            with torch.no_grad():
                outputs_eager = model_eager(
                    past_values=past_values, prediction_length=12, output_hidden_states=True, **covariates
                )
                outputs_sdpa = model_sdpa(
                    past_values=past_values, prediction_length=12, output_hidden_states=True, **covariates
                )
            torch.testing.assert_close(
                outputs_eager.hidden_states[-1].float(),
                outputs_sdpa.hidden_states[-1].float(),
                rtol=tolerance,
                atol=tolerance,
            )
            scale = outputs_sdpa.full_predictions.abs().max().clamp(min=1.0)
            torch.testing.assert_close(
                outputs_eager.full_predictions / scale,
                outputs_sdpa.full_predictions / scale,
                rtol=tolerance,
                atol=tolerance,
            )


@require_torch
@slow
class TimestralModelIntegrationTests(unittest.TestCase):
    @unittest.skip(
        reason="The converted checkpoints are not on the Hub yet. The expected values come from Timestral-209M "
        "converted with `convert_timestral_original_to_hf.py`."
    )
    def test_inference(self):
        model = TimestralModelForPrediction.from_pretrained("mistralai/Timestral-209M").to(torch_device)
        steps = torch.arange(512, dtype=torch.float32, device=torch_device)
        past_values = torch.sin(2 * math.pi * steps / 24)[None, None, :]

        with torch.no_grad():
            outputs = model(past_values=past_values, prediction_length=64)

        self.assertEqual(outputs.mean_predictions.shape, (1, 1, 64))
        self.assertEqual(outputs.full_predictions.shape, (1, 1, 64, 21))
        self.assertEqual((model.config.quantiles[2], model.config.quantiles[18]), (0.1, 0.9))

        # The model carries on the sine wave.
        expected_median_start = torch.tensor(
            [0.866541, 0.707084, 0.497564, 0.253224, -0.003088, -0.260921, -0.500180, -0.709947]
        )
        expected_median_end = torch.tensor([-0.870142, -0.715781, -0.504479, -0.262574])
        expected_quantile_10 = torch.tensor([0.860088, 0.700435, 0.490915, 0.246941])
        expected_quantile_90 = torch.tensor([0.873726, 0.714042, 0.504192, 0.259402])

        median = outputs.mean_predictions[0, 0].cpu()
        quantiles = outputs.full_predictions[0, 0].cpu()
        torch.testing.assert_close(median[:8], expected_median_start, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(median[-4:], expected_median_end, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(quantiles[:4, 2], expected_quantile_10, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(quantiles[:4, 18], expected_quantile_90, rtol=1e-4, atol=1e-4)
