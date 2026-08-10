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
"""Feature extractor for WeatherNext 2.

Turns a physical atmospheric state - a mapping of named variables on a lat/lon grid - into the flat
normalized tensors the model consumes, and turns the model's normalized output back into physical
units. It also owns the two things the network itself has no notion of: the normalization statistics
and the calendar forcings.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)

SECONDS_PER_DAY = 24 * 3600
AVERAGE_DAYS_PER_YEAR = 365.24219


def get_year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
    """Position within the year, in `[0, 1)`."""
    years = seconds_since_epoch / SECONDS_PER_DAY / np.float64(AVERAGE_DAYS_PER_YEAR)
    return np.mod(years, 1.0).astype(np.float32)


def get_day_progress(seconds_since_epoch: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Local position within the day at each longitude, in `[0, 1)`."""
    greenwich = np.mod(seconds_since_epoch, SECONDS_PER_DAY) / SECONDS_PER_DAY
    offsets = np.deg2rad(longitude) / (2 * np.pi)
    return np.mod(greenwich[..., None] + offsets, 1.0).astype(np.float32)


class WeatherNext2Processor(FeatureExtractionMixin):
    r"""
    Constructs a WeatherNext 2 feature extractor.

    Args:
        input_variables (`Sequence[str]`):
            Variables the model consumes, in channel order.
        target_variables (`Sequence[str]`):
            Variables the model predicts, in channel order.
        forcing_variables (`Sequence[str]`):
            Variables supplied for the predicted time step. These are all derived from the clock, so this class
            computes them itself in [`~WeatherNext2Processor.compute_forcings`].
        atmospheric_variables (`Sequence[str]`):
            Variables that carry a pressure-level dimension.
        static_variables (`Sequence[str]`):
            Input variables that do not change over time.
        global_variables (`Sequence[str]`):
            Variables with neither a latitude nor a longitude dimension.
        pressure_levels (`Sequence[int]`):
            Pressure levels in hPa, ascending.
        mean_by_level (`dict[str, float | list[float]]`):
            Per-variable mean used to normalize inputs, and to unnormalize predictions of variables that are not
            themselves inputs.
        stddev_by_level (`dict[str, float | list[float]]`):
            Per-variable standard deviation, used the same way.
        diffs_stddev_by_level (`dict[str, float | list[float]]`):
            Per-variable standard deviation of the one-step difference. Predictions of variables that are also inputs
            are residuals scaled by this.
        nan_filled_variables (`Sequence[str]`, *optional*):
            Variables whose missing values are filled with their mean before normalization, and masked out again in
            the prediction. `sea_surface_temperature` is undefined over land, and the network cannot ingest NaNs.
        num_input_timesteps (`int`, *optional*, defaults to 2):
            Number of past states the model conditions on.
        time_step_hours (`int`, *optional*, defaults to 6):
            Hours advanced by one forward pass.
        grid_latitudes (`int`, *optional*, defaults to 721):
            Number of latitudes, from -90 to 90 inclusive.
        grid_longitudes (`int`, *optional*, defaults to 1440):
            Number of longitudes, from 0 eastwards.
    """

    model_input_names = ["grid_features", "global_features"]

    def __init__(
        self,
        input_variables: Sequence[str],
        target_variables: Sequence[str],
        forcing_variables: Sequence[str],
        atmospheric_variables: Sequence[str],
        static_variables: Sequence[str],
        global_variables: Sequence[str],
        pressure_levels: Sequence[int],
        mean_by_level: Mapping[str, Any],
        stddev_by_level: Mapping[str, Any],
        diffs_stddev_by_level: Mapping[str, Any],
        nan_filled_variables: Sequence[str] | None = None,
        num_input_timesteps: int = 2,
        time_step_hours: int = 6,
        grid_latitudes: int = 721,
        grid_longitudes: int = 1440,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_variables = list(input_variables)
        self.target_variables = list(target_variables)
        self.forcing_variables = list(forcing_variables)
        self.atmospheric_variables = list(atmospheric_variables)
        self.static_variables = list(static_variables)
        self.global_variables = list(global_variables)
        self.pressure_levels = list(pressure_levels)
        self.mean_by_level = dict(mean_by_level)
        self.stddev_by_level = dict(stddev_by_level)
        self.diffs_stddev_by_level = dict(diffs_stddev_by_level)
        self.nan_filled_variables = list(nan_filled_variables or [])
        self.num_input_timesteps = num_input_timesteps
        self.time_step_hours = time_step_hours
        self.grid_latitudes = grid_latitudes
        self.grid_longitudes = grid_longitudes

    # ---------------------------------------------------------------------------------------
    # Layout
    # ---------------------------------------------------------------------------------------

    @property
    def longitudes(self) -> np.ndarray:
        return np.arange(self.grid_longitudes) * (360.0 / self.grid_longitudes)

    def num_levels(self, variable: str) -> int:
        return len(self.pressure_levels) if variable in self.atmospheric_variables else 1

    @property
    def past_time_offsets(self) -> list[int]:
        """Hours, relative to the initialization time, of each conditioning frame."""
        return [-(self.num_input_timesteps - 1 - i) * self.time_step_hours for i in range(self.num_input_timesteps)]

    @property
    def input_channel_layout(self) -> list[tuple[str, int | None, int]]:
        """`(variable, time offset in hours or None, number of levels)` per input channel group."""
        layout: list[tuple[str, int | None, int]] = []
        for variable in self.input_variables:
            if variable in self.static_variables:
                layout.append((variable, None, self.num_levels(variable)))
            else:
                layout.extend((variable, offset, self.num_levels(variable)) for offset in self.past_time_offsets)
        layout.extend(
            (variable, self.time_step_hours, self.num_levels(variable)) for variable in self.forcing_variables
        )
        return layout

    @property
    def mesh_channel_layout(self) -> list[tuple[str, int | None, int]]:
        return [entry for entry in self.input_channel_layout if entry[0] in self.global_variables]

    @property
    def target_channel_layout(self) -> list[tuple[str, int | None, int]]:
        return [(variable, self.time_step_hours, self.num_levels(variable)) for variable in self.target_variables]

    # ---------------------------------------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------------------------------------

    def _statistic(self, table: Mapping[str, Any], variable: str) -> np.ndarray | None:
        """Per-level statistic for a variable, or `None` if it has none.

        The cyclone diagnostics are already dimensionless and carry no statistics; the original
        implementation leaves such variables untouched, so we do the same.
        """
        if variable not in table:
            logger.warning_once(f"No normalization statistic for {variable!r}; leaving it unscaled.")
            return None
        value = np.asarray(table[variable], dtype=np.float32)
        expected = self.num_levels(variable)
        if value.ndim == 0:
            value = value.reshape(1)
        if value.shape != (expected,):
            raise ValueError(f"Statistic for {variable!r} has shape {value.shape}, expected ({expected},).")
        return value

    def normalize(self, values: np.ndarray, variable: str) -> np.ndarray:
        """Maps a variable to roughly zero mean and unit variance, level by level.

        Variables listed in `nan_filled_variables` have their missing values replaced by the mean,
        which lands on exactly zero after normalization.
        """
        mean = self._statistic(self.mean_by_level, variable)
        stddev = self._statistic(self.stddev_by_level, variable)
        normalized = values if mean is None else values - self._broadcast(mean, values)
        if stddev is not None:
            normalized = normalized / self._broadcast(stddev, values)
        if variable in self.nan_filled_variables:
            normalized = np.nan_to_num(normalized, nan=0.0)
        return normalized

    @staticmethod
    def _broadcast(statistic: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Aligns a per-level statistic against `[..., levels, lat, lon]` or `[..., lat, lon]`."""
        if statistic.shape[0] == 1:
            return statistic.reshape([1] * values.ndim)
        return statistic.reshape([1] * (values.ndim - 3) + [-1, 1, 1])

    # ---------------------------------------------------------------------------------------
    # Forcings
    # ---------------------------------------------------------------------------------------

    def compute_forcings(self, seconds_since_epoch: np.ndarray) -> dict[str, np.ndarray]:
        """Clock-derived variables at the given times.

        `year_progress_*` is a single number per time; `day_progress_*` varies with longitude, since
        it encodes local solar time.
        """
        seconds_since_epoch = np.asarray(seconds_since_epoch, dtype=np.int64)
        year = get_year_progress(seconds_since_epoch) * (2 * np.pi)
        day = get_day_progress(seconds_since_epoch, self.longitudes) * (2 * np.pi)
        return {
            "year_progress_sin": np.sin(year),
            "year_progress_cos": np.cos(year),
            "day_progress_sin": np.sin(day),
            "day_progress_cos": np.cos(day),
        }

    # ---------------------------------------------------------------------------------------
    # Encoding / decoding
    # ---------------------------------------------------------------------------------------

    def _field(self, state: Mapping[str, Any], variable: str, time_index: int | None) -> np.ndarray:
        """Extracts `[batch, levels, lat, lon]` for one variable at one conditioning frame."""
        values = np.asarray(state[variable], dtype=np.float32)
        if variable in self.static_variables:
            while values.ndim < 4:
                values = values[None]
            return np.broadcast_to(values, (values.shape[0], 1, self.grid_latitudes, self.grid_longitudes))

        values = values[:, time_index]
        if variable in self.global_variables:
            # [batch] -> broadcast over the whole grid.
            return np.broadcast_to(
                values.reshape(-1, 1, 1, 1), (values.shape[0], 1, self.grid_latitudes, self.grid_longitudes)
            )
        if values.ndim == 3:
            # [batch, lat, lon] or [batch, lon] for the longitude-only forcings.
            if values.shape[-2:] == (self.grid_latitudes, self.grid_longitudes):
                return values[:, None]
        if values.ndim == 2 and values.shape[-1] == self.grid_longitudes:
            return np.broadcast_to(
                values[:, None, None], (values.shape[0], 1, self.grid_latitudes, self.grid_longitudes)
            )
        return values

    def __call__(
        self,
        state: Mapping[str, Any],
        seconds_since_epoch: np.ndarray | None = None,
        forcings: Mapping[str, Any] | None = None,
        return_tensors: str | TensorType | None = TensorType.PYTORCH,
    ) -> BatchFeature:
        """Encodes a physical state into model inputs.

        Args:
            state (`Mapping[str, array]`):
                Physical values keyed by variable name. Time-varying variables have shape
                `[batch, num_input_timesteps, (levels,) latitudes, longitudes]`, static variables `[latitudes,
                longitudes]`, and variables with no spatial extent `[batch, num_input_timesteps]`. Longitude-only
                variables such as `day_progress_sin` may be `[batch, num_input_timesteps, longitudes]`.
            seconds_since_epoch (`np.ndarray` of shape `(batch,)`, *optional*):
                Valid time of the *predicted* step, used to derive the forcings. Required unless `forcings` is given.
            forcings (`Mapping[str, array]`, *optional*):
                Precomputed forcings for the predicted step, overriding `seconds_since_epoch`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `"pt"`):
                Framework of the returned tensors.

        Returns:
            [`BatchFeature`] with `grid_features` of shape `[batch, channels, latitudes, longitudes]` and
            `global_features` of shape `[batch, global_channels]`.
        """
        if forcings is None:
            if seconds_since_epoch is None:
                raise ValueError("Pass either `seconds_since_epoch` or `forcings`.")
            forcings = self.compute_forcings(np.asarray(seconds_since_epoch))
        forcings = {name: np.asarray(values, dtype=np.float32)[:, None] for name, values in forcings.items()}

        grid_channels = []
        global_channels = []
        for variable, offset, _ in self.input_channel_layout:
            if offset is not None and offset > 0:
                field = self._field(forcings, variable, 0)
            elif offset is None:
                field = self._field(state, variable, None)
            else:
                field = self._field(state, variable, self.past_time_offsets.index(offset))
            field = self.normalize(field, variable)
            grid_channels.append(field)
            if variable in self.global_variables:
                global_channels.append(field[..., 0, 0])

        data = {
            "grid_features": np.concatenate(grid_channels, axis=1),
            "global_features": np.concatenate(global_channels, axis=1),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def postprocess(self, prediction: Any, state: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
        """Decodes the model's normalized output into physical units.

        Variables that also appear in the inputs are predicted as normalized residuals and need the last
        conditioning frame added back, so `state` is required whenever any target is also an input.

        Args:
            prediction (`array` of shape `(batch, channels, latitudes, longitudes)`):
                [`WeatherNext2ForecastOutput.prediction`].
            state (`Mapping[str, array]`, *optional*):
                The same state that was passed to `__call__`.

        Returns:
            `dict[str, np.ndarray]`: physical values keyed by variable, shaped `[batch, (levels,) lat, lon]`.
        """
        prediction = np.asarray(
            prediction.detach().cpu() if hasattr(prediction, "detach") else prediction, dtype=np.float32
        )
        outputs: dict[str, np.ndarray] = {}
        offset = 0
        for variable, _, levels in self.target_channel_layout:
            values = prediction[:, offset : offset + levels]
            offset += levels
            if state is not None and variable in state and variable not in self.static_variables:
                residual_scale = self._statistic(self.diffs_stddev_by_level, variable)
                last_frame = np.asarray(state[variable], dtype=np.float32)[:, -1]
                if last_frame.ndim == 3:
                    last_frame = last_frame[:, None]
                if residual_scale is not None:
                    values = values * self._broadcast(residual_scale, values)
                values = values + last_frame
            else:
                mean = self._statistic(self.mean_by_level, variable)
                stddev = self._statistic(self.stddev_by_level, variable)
                if stddev is not None:
                    values = values * self._broadcast(stddev, values)
                if mean is not None:
                    values = values + self._broadcast(mean, values)
            if state is not None and variable in self.nan_filled_variables and variable in state:
                # The land mask is constant across the conditioning frames, so any frame will do.
                missing = np.isnan(np.asarray(state[variable], dtype=np.float32)).any(axis=1)
                if missing.ndim == values.ndim - 1:
                    missing = missing[:, None]
                values = np.where(missing, np.nan, values)
            outputs[variable] = values[:, 0] if levels == 1 else values
        return outputs

    def advance_state(
        self,
        state: Mapping[str, Any],
        forecast: Mapping[str, np.ndarray],
        seconds_since_epoch: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Builds the conditioning state for the next autoregressive step.

        Drops the oldest frame, appends the forecast, and recomputes the clock variables. Targets that are not also
        inputs (precipitation, the cyclone diagnostics) are simply discarded.
        """
        next_state: dict[str, np.ndarray] = {}
        forcings = self.compute_forcings(np.asarray(seconds_since_epoch))
        for variable in self.input_variables:
            if variable in self.static_variables:
                next_state[variable] = np.asarray(state[variable], dtype=np.float32)
            elif variable in forcings:
                previous = np.asarray(state[variable], dtype=np.float32)
                latest = np.asarray(forcings[variable], dtype=np.float32)[:, None]
                next_state[variable] = np.concatenate([previous[:, 1:], latest], axis=1)
            else:
                previous = np.asarray(state[variable], dtype=np.float32)
                next_state[variable] = np.concatenate([previous[:, 1:], forecast[variable][:, None]], axis=1)
        return next_state


__all__ = ["WeatherNext2Processor"]
