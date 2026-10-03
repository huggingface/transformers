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
        # `total_precipitation_6hr` is a target only, so it is unnormalized rather than residual.
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
        """A physical atmospheric state, shaped the way the docs describe it."""
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
        # Sea surface temperature is undefined over land; the extractor has to fill and re-mask it.
        state["sea_surface_temperature"][..., 0, :] = np.nan
        # The clock variables are part of the conditioning state, not just of the predicted step.
        past = self.prepare_times(batch_size)[:, None] + np.array(
            [offset * 3600 for offset in [-self.time_step_hours, 0]], dtype=np.int64
        )
        for name, values in self.forcings_for(past).items():
            state[name] = values
        return {name: values.astype(np.float32) for name, values in state.items()}

    def forcings_for(self, times):
        """Clock variables for a `[batch, frames]` array of times, shaped as conditioning frames."""
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


class WeatherNext2FeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):
    feature_extraction_class = WeatherNext2FeatureExtractor

    def setUp(self):
        self.feat_extract_tester = WeatherNext2FeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feat_extract_tester.prepare_feat_extract_dict()

    def test_init_without_params(self):
        """The variable lists and normalization statistics come from a checkpoint, so there is nothing to default to."""
        with self.assertRaises(TypeError):
            self.feature_extraction_class()

    def test_grid_coordinates_describe_the_fields(self):
        """The coordinates the fields are laid out on, which anything area-weighting them needs."""
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        latitudes, longitudes = extractor.latitudes, extractor.longitudes
        self.assertEqual(latitudes.shape, (extractor.grid_latitudes,))
        self.assertEqual(longitudes.shape, (extractor.grid_longitudes,))
        # Latitudes run pole to pole inclusive, longitudes eastwards from zero and stop short of 360.
        self.assertEqual((latitudes[0], latitudes[-1]), (-90.0, 90.0))
        self.assertEqual(longitudes[0], 0.0)
        self.assertLess(longitudes[-1], 360.0)
        for values in (latitudes, longitudes):
            np.testing.assert_allclose(np.diff(values), np.diff(values)[0])

    def test_call_returns_expected_shapes(self):
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        inputs = extractor(tester.prepare_state(), seconds_since_epoch=tester.prepare_times(), return_tensors="np")

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

    def test_normalize_round_trips(self):
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        values = np.array([[[[300.0]]]], dtype=np.float32)
        normalized = extractor.normalize(values, "2m_temperature")
        self.assertAlmostEqual(normalized.item(), (300.0 - 280.0) / 20.0, places=5)

    def test_forcings_are_periodic_and_longitude_dependent(self):
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        times = np.array([0, 365 * 24 * 3600], dtype=np.int64)
        forcings = extractor.compute_forcings(times)

        self.assertEqual(forcings["year_progress_sin"].shape, (2,))
        self.assertEqual(forcings["day_progress_sin"].shape, (2, extractor.grid_longitudes))
        # sin^2 + cos^2 = 1 for both pairs.
        for prefix, computed in (("year", forcings), ("day", forcings)):
            total = computed[f"{prefix}_progress_sin"] ** 2 + computed[f"{prefix}_progress_cos"] ** 2
            self.assertTrue(np.allclose(total, 1.0, atol=1e-5))
        # Local solar time differs between longitudes.
        day = forcings["day_progress_sin"]
        self.assertFalse(np.allclose(day[:, 0], day[:, extractor.grid_longitudes // 2]))

    def test_postprocess_inverts_the_residual_encoding(self):
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        state = tester.prepare_state()

        channels = sum(levels for _, _, levels in extractor.target_channel_layout)
        prediction = np.zeros((tester.batch_size, channels, tester.grid_latitudes, tester.grid_longitudes), np.float32)
        forecast = extractor.postprocess(prediction, state)

        # A zero residual leaves a variable that is also an input at its last conditioning frame.
        np.testing.assert_allclose(forecast["2m_temperature"], state["2m_temperature"][:, -1], rtol=1e-5)
        self.assertEqual(forecast["temperature"].shape, (tester.batch_size, len(tester.pressure_levels), 5, 8))
        # A target that is not an input is unnormalized with the mean/stddev instead.
        np.testing.assert_allclose(
            forecast["total_precipitation_6hr"],
            np.full_like(forecast["total_precipitation_6hr"], 0.001),
            atol=1e-8,
        )

    def test_postprocess_requires_state_for_residual_targets(self):
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        channels = sum(levels for _, _, levels in extractor.target_channel_layout)
        prediction = np.zeros((tester.batch_size, channels, tester.grid_latitudes, tester.grid_longitudes), np.float32)

        with self.assertRaisesRegex(ValueError, "state.*residual targets"):
            extractor.postprocess(prediction)

    def test_postprocess_rejects_the_wrong_shape(self):
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        prediction = np.zeros((tester.batch_size, 1, tester.grid_latitudes, tester.grid_longitudes), np.float32)

        with self.assertRaisesRegex(ValueError, "prediction.*shape"):
            extractor.postprocess(prediction, tester.prepare_state())

    def test_postprocess_restores_the_sea_surface_temperature_mask(self):
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        state = tester.prepare_state()

        channels = sum(levels for _, _, levels in extractor.target_channel_layout)
        prediction = np.zeros((tester.batch_size, channels, tester.grid_latitudes, tester.grid_longitudes), np.float32)
        forecast = extractor.postprocess(prediction, state)

        sst = forecast["sea_surface_temperature"]
        self.assertTrue(np.isnan(sst[:, 0, :]).all())
        self.assertFalse(np.isnan(sst[:, 1:, :]).any())

    def test_advance_state_shifts_the_time_window(self):
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        state = tester.prepare_state()
        times = tester.prepare_times()

        forecast = {name: np.zeros_like(state[name][:, -1]) for name in tester.surface_variables}
        forecast["temperature"] = np.zeros_like(state["temperature"][:, -1])
        next_state = extractor.advance_state(state, forecast, times + tester.time_step_hours * 3600)

        # Targets that are not inputs are dropped, statics are carried through unchanged.
        self.assertNotIn("total_precipitation_6hr", next_state)
        np.testing.assert_allclose(next_state["land_sea_mask"], state["land_sea_mask"])
        for name in tester.surface_variables + tester.atmospheric_variables:
            self.assertEqual(next_state[name].shape, state[name].shape)
            # The oldest frame is gone: frame 0 of the new state is frame 1 of the old one.
            np.testing.assert_allclose(next_state[name][:, 0], state[name][:, 1], equal_nan=True)
            np.testing.assert_allclose(next_state[name][:, -1], forecast[name], equal_nan=True)

    def test_advance_state_recomputes_the_clock(self):
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        state = tester.prepare_state()
        times = tester.prepare_times()
        next_times = times + tester.time_step_hours * 3600
        forecast = {name: np.zeros_like(state[name][:, -1]) for name in tester.surface_variables}
        forecast["temperature"] = np.zeros_like(state["temperature"][:, -1])
        next_state = extractor.advance_state(state, forecast, next_times)

        expected = extractor.compute_forcings(next_times)
        for name in tester.forcing_variables:
            np.testing.assert_allclose(next_state[name][:, -1], expected[name], rtol=1e-5)

    def test_batching_is_independent(self):
        """A batched call must give each member exactly what it would get on its own."""
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        state = tester.prepare_state()
        times = tester.prepare_times()

        batched = extractor(state, seconds_since_epoch=times, return_tensors="np")
        for member in range(tester.batch_size):
            single_state = {
                name: values if name in tester.static_variables else values[member : member + 1]
                for name, values in state.items()
            }
            single = extractor(single_state, seconds_since_epoch=times[member : member + 1], return_tensors="np")
            for key in ("grid_features", "global_features"):
                np.testing.assert_allclose(single[key][0], batched[key][member], rtol=1e-6, atol=1e-6)

    @require_torch
    def test_tensor_inputs_are_returned_as_tensors(self):
        """Passing tensors keeps every stage on their device, so a rollout never round-trips to the host."""
        import torch

        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        state = tester.prepare_state()
        times = tester.prepare_times()

        tensor_state = {name: torch.from_numpy(values) for name, values in state.items()}
        numpy_inputs = extractor(state, seconds_since_epoch=times, return_tensors="np")
        tensor_inputs = extractor(tensor_state, seconds_since_epoch=times)
        for key in ("grid_features", "global_features"):
            self.assertIsInstance(tensor_inputs[key], torch.Tensor)
            np.testing.assert_allclose(tensor_inputs[key].numpy(), numpy_inputs[key], rtol=1e-6, atol=1e-6)

        channels = sum(levels for _, _, levels in extractor.target_channel_layout)
        prediction = (
            np.random.default_rng(0)
            .standard_normal((tester.batch_size, channels, tester.grid_latitudes, tester.grid_longitudes))
            .astype(np.float32)
        )

        numpy_forecast = extractor.postprocess(prediction, state)
        tensor_forecast = extractor.postprocess(torch.from_numpy(prediction), tensor_state)
        for name, values in numpy_forecast.items():
            self.assertIsInstance(tensor_forecast[name], torch.Tensor)
            np.testing.assert_allclose(tensor_forecast[name].numpy(), values, rtol=1e-6, atol=1e-6, equal_nan=True)

        next_times = times + tester.time_step_hours * 3600
        numpy_next = extractor.advance_state(state, numpy_forecast, next_times)
        tensor_next = extractor.advance_state(tensor_state, tensor_forecast, next_times)
        for name, values in numpy_next.items():
            self.assertIsInstance(tensor_next[name], torch.Tensor)
            np.testing.assert_allclose(tensor_next[name].numpy(), values, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_rollout_keeps_the_clock_aligned_with_the_frames(self):
        """Each frame's clock variables must describe the time that frame is valid at.

        `advance_state` stamps the frame it appends, so it takes the time the forecast is valid at - the same value
        `__call__` was given - and the caller advances the clock afterwards. Doing it the other way round leaves the
        physical fields a step behind their own clock, which no shape check would catch.
        """
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        step_seconds = tester.time_step_hours * 3600

        state = tester.prepare_state()
        valid_time = tester.prepare_times() + step_seconds

        for _ in range(3):
            forecast = {name: np.zeros_like(state[name][:, -1]) for name in tester.surface_variables}
            forecast["temperature"] = np.zeros_like(state["temperature"][:, -1])
            state = extractor.advance_state(state, forecast, valid_time)

            frame_times = np.stack(
                [valid_time - offset * step_seconds for offset in reversed(range(tester.num_input_timesteps))], axis=1
            )
            for index in range(tester.num_input_timesteps):
                expected = extractor.compute_forcings(frame_times[:, index])
                for name in tester.forcing_variables:
                    np.testing.assert_allclose(state[name][:, index], expected[name], rtol=1e-5, atol=1e-5)

            valid_time = valid_time + step_seconds

    def test_call_rejects_a_missing_time(self):
        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        with self.assertRaises(ValueError):
            extractor(tester.prepare_state())

    @require_torch
    def test_call_returns_torch_tensors(self):
        import torch

        tester = self.feat_extract_tester
        extractor = self.feature_extraction_class(**self.feat_extract_dict)
        inputs = extractor(tester.prepare_state(), seconds_since_epoch=tester.prepare_times())
        self.assertIsInstance(inputs["grid_features"], torch.Tensor)
        self.assertEqual(inputs["grid_features"].dtype, torch.float32)
