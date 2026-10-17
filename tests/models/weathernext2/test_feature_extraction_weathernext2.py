# Copyright 2026 Google DeepMind and The HuggingFace Inc. team. All rights reserved.
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

import tempfile
import unittest

import numpy as np

from transformers import WeatherNext2FeatureExtractor
from transformers.testing_utils import require_torch

from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin


class WeatherNext2FeatureExtractionTester:
    """A miniature version of the released feature extractor, on a coarse grid."""

    def __init__(
        self,
        parent,
        batch_size=3,
        grid_latitudes=5,
        grid_longitudes=8,
        pressure_levels=(500, 850),
        num_input_timesteps=2,
        time_step_hours=6,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.grid_latitudes = grid_latitudes
        self.grid_longitudes = grid_longitudes
        self.pressure_levels = list(pressure_levels)
        self.num_input_timesteps = num_input_timesteps
        self.time_step_hours = time_step_hours

        self.atmospheric_variables = ["temperature"]
        self.static_variables = ["land_sea_mask"]
        self.global_variables = ["year_progress_sin", "year_progress_cos"]
        self.forcing_variables = ["year_progress_sin", "year_progress_cos", "day_progress_sin", "day_progress_cos"]
        self.surface_variables = ["2m_temperature", "sea_surface_temperature"]
        self.input_variables = self.surface_variables + self.atmospheric_variables + self.static_variables
        self.target_variables = self.surface_variables + self.atmospheric_variables + ["total_precipitation_6hr"]

    def prepare_feat_extract_dict(self):
        levels = len(self.pressure_levels)
        return {
            "input_variables": self.input_variables + self.forcing_variables,
            "target_variables": self.target_variables,
            "forcing_variables": self.forcing_variables,
            "atmospheric_variables": self.atmospheric_variables,
            "static_variables": self.static_variables,
            "global_variables": self.global_variables,
            "pressure_levels": self.pressure_levels,
            "mean_by_level": {
                "2m_temperature": 280.0,
                "sea_surface_temperature": 290.0,
                "temperature": [250.0] * levels,
                "land_sea_mask": 0.3,
                "total_precipitation_6hr": 0.001,
            },
            "stddev_by_level": {
                "2m_temperature": 20.0,
                "sea_surface_temperature": 10.0,
                "temperature": [15.0] * levels,
                "land_sea_mask": 0.5,
                "total_precipitation_6hr": 0.002,
            },
            "diffs_stddev_by_level": {
                "2m_temperature": 2.0,
                "sea_surface_temperature": 0.5,
                "temperature": [1.5] * levels,
            },
            "nan_fill_values": {"sea_surface_temperature": 269.48291015625},
            "num_input_timesteps": self.num_input_timesteps,
            "time_step_hours": self.time_step_hours,
            "grid_latitudes": self.grid_latitudes,
            "grid_longitudes": self.grid_longitudes,
        }

    def prepare_state(self, batch_size=None, seed=0):
        batch_size = self.batch_size if batch_size is None else batch_size
        generator = np.random.default_rng(seed)
        shape = (batch_size, self.num_input_timesteps, self.grid_latitudes, self.grid_longitudes)
        state = {
            "2m_temperature": 280.0 + 20.0 * generator.standard_normal(shape),
            "sea_surface_temperature": 290.0 + 10.0 * generator.standard_normal(shape),
            "temperature": 250.0
            + 15.0
            * generator.standard_normal((batch_size, self.num_input_timesteps, len(self.pressure_levels), *shape[2:])),
            "land_sea_mask": generator.random((self.grid_latitudes, self.grid_longitudes)),
        }
        state["sea_surface_temperature"][..., 0, :] = np.nan
        past = self.prepare_times(batch_size)[:, None] + np.array(
            [offset * 3600 for offset in [-self.time_step_hours, 0]], dtype=np.int64
        )
        for name, values in self.forcings_for(past).items():
            state[name] = values
        return {name: values.astype(np.float32) for name, values in state.items()}

    def forcings_for(self, times):
        greenwich = np.mod(times, 24 * 3600) / (24 * 3600)
        longitudes = np.arange(self.grid_longitudes) * (360.0 / self.grid_longitudes)
        year = np.mod(times / (24 * 3600) / 365.24219, 1.0) * 2 * np.pi
        day = np.mod(greenwich[..., None] + np.deg2rad(longitudes) / (2 * np.pi), 1.0) * 2 * np.pi
        return {
            "year_progress_sin": np.sin(year),
            "year_progress_cos": np.cos(year),
            "day_progress_sin": np.sin(day),
            "day_progress_cos": np.cos(day),
        }

    def prepare_times(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        start = np.datetime64("2024-10-07T06:00:00").astype("datetime64[s]").astype(np.int64)
        return start + np.arange(batch_size, dtype=np.int64) * 6 * 3600


@require_torch
class WeatherNext2FeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):
    feature_extraction_class = WeatherNext2FeatureExtractor

    def setUp(self):
        self.feat_extract_tester = WeatherNext2FeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feat_extract_tester.prepare_feat_extract_dict()

    @unittest.skip(reason="WeatherNext2 requires checkpoint normalization statistics.")
    def test_init_without_params(self):
        pass

    def test_call(self):
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        state = tester.prepare_state()
        inputs = extractor(state, seconds_since_epoch=tester.prepare_times(), return_tensors="np")

        channels = sum(levels for _, _, levels in extractor.input_channel_layout)
        self.assertEqual(
            inputs["grid_features"].shape,
            (tester.batch_size, channels, tester.grid_latitudes, tester.grid_longitudes),
        )
        self.assertEqual(
            inputs["global_features"].shape,
            (tester.batch_size, sum(levels for _, _, levels in extractor.mesh_channel_layout)),
        )
        self.assertFalse(np.isnan(inputs["grid_features"]).any())
        self.assertEqual(inputs["noise"].shape, (tester.batch_size, extractor.noise_channels))
        np.testing.assert_allclose(inputs["grid_features"][:, :2], (state["2m_temperature"] - 280.0) / 20.0)

    def test_postprocess_inverts_the_residual_encoding(self):
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        state = tester.prepare_state()

        channels = sum(levels for _, _, levels in extractor.target_channel_layout)
        prediction = np.zeros((tester.batch_size, channels, tester.grid_latitudes, tester.grid_longitudes), np.float32)
        forecast = extractor.postprocess(prediction, state)

        np.testing.assert_allclose(forecast["2m_temperature"], state["2m_temperature"][:, -1], rtol=1e-5)
        self.assertEqual(forecast["temperature"].shape, (tester.batch_size, len(tester.pressure_levels), 5, 8))
        # Precipitation is a target only, not a residual.
        np.testing.assert_allclose(
            forecast["total_precipitation_6hr"],
            np.full_like(forecast["total_precipitation_6hr"], 0.001),
            atol=1e-8,
        )

        sst = forecast["sea_surface_temperature"]
        self.assertTrue(np.isnan(sst[:, 0, :]).all())
        self.assertFalse(np.isnan(sst[:, 1:, :]).any())

    def test_batching_is_independent(self):
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        state = tester.prepare_state()
        times = tester.prepare_times()

        batched = extractor(state, seconds_since_epoch=times, return_tensors="np")
        ensemble = extractor(state, seconds_since_epoch=times, num_members=3, return_tensors="np")
        self.assertEqual(ensemble["noise"].shape, (tester.batch_size * 3, extractor.noise_channels))
        self.assertFalse(np.array_equal(ensemble["noise"][0], ensemble["noise"][1]))
        for key in ("grid_features", "global_features"):
            np.testing.assert_array_equal(ensemble[key], np.repeat(batched[key], 3, axis=0))
        for member in range(tester.batch_size):
            single_state = {
                name: values if name in tester.static_variables else values[member : member + 1]
                for name, values in state.items()
            }
            single = extractor(single_state, seconds_since_epoch=times[member : member + 1], return_tensors="np")
            for key in ("grid_features", "global_features"):
                np.testing.assert_allclose(single[key][0], batched[key][member], rtol=1e-6, atol=1e-6)

    def test_numpy_torch_inputs(self):
        import torch

        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        state = tester.prepare_state()
        times = tester.prepare_times()
        tensor_state = {name: torch.from_numpy(values) for name, values in state.items()}

        from_numpy = extractor(state, seconds_since_epoch=times, generator=torch.Generator().manual_seed(0))
        from_tensors = extractor(
            tensor_state, seconds_since_epoch=torch.from_numpy(times), generator=torch.Generator().manual_seed(0)
        )
        for key in ("grid_features", "global_features", "noise"):
            self.assertIsInstance(from_numpy[key], torch.Tensor)
            torch.testing.assert_close(from_numpy[key], from_tensors[key])

        channels = sum(levels for _, _, levels in extractor.target_channel_layout)
        prediction = torch.randn(tester.batch_size, channels, tester.grid_latitudes, tester.grid_longitudes)
        forecast = extractor.postprocess(prediction.numpy(), state)
        tensor_forecast = extractor.postprocess(prediction, tensor_state)
        for name, values in forecast.items():
            self.assertIsInstance(values, torch.Tensor)
            torch.testing.assert_close(values, tensor_forecast[name], equal_nan=True)

        next_times = times + tester.time_step_hours * 3600
        next_state = extractor.advance_state(state, forecast, next_times)
        tensor_next = extractor.advance_state(tensor_state, tensor_forecast, torch.from_numpy(next_times))
        for name, values in next_state.items():
            self.assertIsInstance(values, torch.Tensor)
            torch.testing.assert_close(values, tensor_next[name], equal_nan=True)

    def test_rollout_keeps_the_clock_aligned_with_the_frames(self):
        """Physical fields and clock forcings must advance together."""
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        step_seconds = tester.time_step_hours * 3600

        state = tester.prepare_state()
        valid_time = tester.prepare_times() + step_seconds

        for _ in range(3):
            forecast = {name: np.zeros_like(state[name][:, -1]) for name in tester.surface_variables}
            forecast["temperature"] = np.zeros_like(state["temperature"][:, -1])
            forecast["total_precipitation_6hr"] = np.zeros_like(forecast["2m_temperature"])
            next_state = extractor.advance_state(state, forecast, valid_time)

            self.assertNotIn("total_precipitation_6hr", next_state)
            np.testing.assert_array_equal(next_state["land_sea_mask"], state["land_sea_mask"])
            for name in tester.surface_variables + tester.atmospheric_variables:
                np.testing.assert_allclose(next_state[name][:, 0], state[name][:, 1], equal_nan=True)
                np.testing.assert_allclose(next_state[name][:, -1], forecast[name], equal_nan=True)
            state = next_state

            frame_times = np.stack(
                [valid_time - offset * step_seconds for offset in reversed(range(tester.num_input_timesteps))], axis=1
            )
            for index in range(tester.num_input_timesteps):
                expected = tester.forcings_for(frame_times[:, index])
                for name in tester.forcing_variables:
                    np.testing.assert_allclose(state[name][:, index], expected[name], rtol=1e-5, atol=1e-5)

            valid_time = valid_time + step_seconds

    def test_attention_mask(self):
        import torch

        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        attention_mask = torch.ones(2, 1, 4, 12, dtype=torch.bool)
        inputs = extractor(
            tester.prepare_state(), seconds_since_epoch=tester.prepare_times(), attention_mask=attention_mask
        )
        self.assertIs(inputs["attention_mask"], attention_mask)
        inputs = inputs.to(dtype=torch.bfloat16)
        self.assertEqual(inputs["grid_features"].dtype, torch.bfloat16)
        self.assertIs(inputs["attention_mask"], attention_mask)
        extractor.attention_mask = attention_mask
        with tempfile.TemporaryDirectory() as directory:
            extractor.save_pretrained(directory)
            restored = self.feature_extraction_class.from_pretrained(directory)
        torch.testing.assert_close(restored.attention_mask, attention_mask)
        self.assertNotIn("attention_mask", restored.to_dict())
