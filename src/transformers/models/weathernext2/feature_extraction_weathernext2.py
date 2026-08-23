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

The arithmetic runs in torch rather than numpy so that an autoregressive rollout can stay on the
accelerator: numpy inputs are adopted with `torch.as_tensor`, which does not copy, and results are
handed back as numpy again unless the caller passed tensors.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...utils import TensorType, is_torch_available, logging, requires_backends


if is_torch_available():
    import torch


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


class WeatherNext2FeatureExtractor(FeatureExtractionMixin):
    r"""
    Constructs a WeatherNext 2 feature extractor.

    Args:
        input_variables (`Sequence[str]`):
            Variables the model consumes, in channel order.
        target_variables (`Sequence[str]`):
            Variables the model predicts, in channel order.
        forcing_variables (`Sequence[str]`):
            Variables supplied for the predicted time step. These are all derived from the clock, so this class
            computes them itself in [`~WeatherNext2FeatureExtractor.compute_forcings`].
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
        nan_fill_values (`dict[str, float]`, *optional*):
            Value substituted for missing data, per variable, before normalization; the prediction is masked out
            again wherever the input was missing. `sea_surface_temperature` is undefined over land and the network
            cannot ingest NaNs, so it is filled with a fixed temperature rather than with its own mean.
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
        nan_fill_values: Mapping[str, float] | None = None,
        num_input_timesteps: int = 2,
        time_step_hours: int = 6,
        grid_latitudes: int = 721,
        grid_longitudes: int = 1440,
        **kwargs,
    ):
        # The arithmetic here is all torch, so that a rollout can stay on the accelerator.
        requires_backends(self, ["torch"])
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
        self.nan_fill_values = dict(nan_fill_values or {})
        self.num_input_timesteps = num_input_timesteps
        self.time_step_hours = time_step_hours
        self.grid_latitudes = grid_latitudes
        self.grid_longitudes = grid_longitudes

    @property
    def latitudes(self) -> np.ndarray:
        return np.linspace(-90.0, 90.0, self.grid_latitudes)

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

    def _statistic(self, table: Mapping[str, Any], variable: str) -> "torch.Tensor | None":
        """Per-level statistic for a variable, or `None` if it has none.

        The cyclone diagnostics are already dimensionless and carry no statistics; the original
        implementation leaves such variables untouched, so we do the same.
        """
        if variable not in table:
            logger.warning_once(f"No normalization statistic for {variable!r}; leaving it unscaled.")
            return None
        value = torch.as_tensor(table[variable], dtype=torch.float32).reshape(-1)
        expected = self.num_levels(variable)
        if value.shape != (expected,):
            raise ValueError(f"Statistic for {variable!r} has shape {tuple(value.shape)}, expected ({expected},).")
        return value

    def normalize(self, values: np.ndarray, variable: str) -> np.ndarray:
        """Maps a variable to roughly zero mean and unit variance, level by level.

        Missing values are replaced first, in physical units, so that the substituted value lands
        wherever the statistics put it rather than on zero.
        """
        values = torch.as_tensor(values, dtype=torch.float32)
        if variable in self.nan_fill_values:
            values = torch.nan_to_num(values, nan=self.nan_fill_values[variable])
        mean = self._statistic(self.mean_by_level, variable)
        stddev = self._statistic(self.stddev_by_level, variable)
        normalized = values if mean is None else values - self._broadcast(mean, values)
        if stddev is not None:
            normalized = normalized / self._broadcast(stddev, values)
        return normalized

    @staticmethod
    def _broadcast(statistic: "torch.Tensor", values: "torch.Tensor") -> "torch.Tensor":
        """Aligns a per-level statistic against `[..., levels, lat, lon]` or `[..., lat, lon]`."""
        shape = [1] * values.ndim if statistic.shape[0] == 1 else [1] * (values.ndim - 3) + [-1, 1, 1]
        return statistic.to(values.device).reshape(shape)

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

    def _batch_size(self, state: Mapping[str, Any], forcings: Mapping[str, Any]) -> int:
        """Number of ensemble members in a call, taken from the first variable that carries a batch axis.

        Static fields have no batch axis, so they cannot be used; every other input does, and they must agree.
        """
        sizes = {
            torch.as_tensor(values).shape[0]
            for source in (state, forcings)
            for name, values in source.items()
            if name not in self.static_variables and torch.as_tensor(values).ndim > 0
        }
        if len(sizes) != 1:
            raise ValueError(f"Inconsistent batch sizes across the inputs: {sorted(sizes)}.")
        return sizes.pop()

    def _field(
        self, state: Mapping[str, Any], variable: str, time_index: int | None, batch_size: int = 1
    ) -> "torch.Tensor":
        """Extracts `[batch, levels, lat, lon]` for one variable at one conditioning frame."""
        values = torch.as_tensor(state[variable], dtype=torch.float32)
        if variable in self.static_variables:
            # Static fields carry no batch axis of their own, so they are shared by every member.
            values = values.reshape(-1, 1, self.grid_latitudes, self.grid_longitudes)
            return values.expand(batch_size, 1, self.grid_latitudes, self.grid_longitudes)

        values = values[:, time_index]
        if variable in self.global_variables:
            # [batch] -> broadcast over the whole grid.
            return values.reshape(-1, 1, 1, 1).expand(values.shape[0], 1, self.grid_latitudes, self.grid_longitudes)
        if values.ndim == 3:
            # [batch, lat, lon] or [batch, lon] for the longitude-only forcings.
            if values.shape[-2:] == (self.grid_latitudes, self.grid_longitudes):
                return values[:, None]
        if values.ndim == 2 and values.shape[-1] == self.grid_longitudes:
            return values[:, None, None].expand(values.shape[0], 1, self.grid_latitudes, self.grid_longitudes)
        return values

    def __call__(
        self,
        state: Mapping[str, Any],
        seconds_since_epoch: np.ndarray | None = None,
        forcings: Mapping[str, Any] | None = None,
        return_tensors: str | TensorType | None = TensorType.PYTORCH,
        device: "torch.device | str | None" = None,
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
            device (`torch.device` or `str`, *optional*):
                Device to encode on. Defaults to the device of `state` when it holds tensors, and to the CPU
                otherwise, so an autoregressive rollout that starts on an accelerator stays there.

        Returns:
            [`BatchFeature`] with `grid_features` of shape `[batch, channels, latitudes, longitudes]` and
            `global_features` of shape `[batch, global_channels]`.
        """
        if forcings is None:
            if seconds_since_epoch is None:
                raise ValueError("Pass either `seconds_since_epoch` or `forcings`.")
            forcings = self.compute_forcings(np.asarray(seconds_since_epoch))
        forcings = {name: torch.as_tensor(values, dtype=torch.float32)[:, None] for name, values in forcings.items()}

        # The clock forcings and any static fields are built on the host, so every channel is moved to
        # one device before they are concatenated.
        if device is None:
            device = next(
                (values.device for values in state.values() if isinstance(values, torch.Tensor)),
                torch.device("cpu"),
            )
        batch_size = self._batch_size(state, forcings)
        grid_channels = []
        global_channels = []
        for variable, offset, _ in self.input_channel_layout:
            if offset is not None and offset > 0:
                field = self._field(forcings, variable, 0, batch_size)
            elif offset is None:
                field = self._field(state, variable, None, batch_size)
            else:
                field = self._field(state, variable, self.past_time_offsets.index(offset), batch_size)
            field = self.normalize(field.to(device), variable)
            grid_channels.append(field)
            if variable in self.global_variables:
                global_channels.append(field[..., 0, 0])

        data = {
            "grid_features": torch.cat(grid_channels, dim=1),
            "global_features": torch.cat(global_channels, dim=1),
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
                The same state that was passed to `__call__`. Required when an input variable is
                also a forecast target because those targets are decoded as residuals.

        Returns:
            `dict[str, array]`: physical values keyed by variable, shaped `[batch, (levels,) lat, lon]`, as
            tensors on the device of `prediction` if it was a tensor and as numpy arrays otherwise.
        """
        as_numpy = not isinstance(prediction, torch.Tensor)
        prediction = torch.as_tensor(prediction, dtype=torch.float32).detach()
        expected_shape = (
            sum(levels for _, _, levels in self.target_channel_layout),
            self.grid_latitudes,
            self.grid_longitudes,
        )
        if prediction.ndim != 4 or tuple(prediction.shape[1:]) != expected_shape:
            raise ValueError(
                f"`prediction` has shape {tuple(prediction.shape)}, expected "
                f"(batch_size, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]})."
            )
        residual_targets = set(self.target_variables).intersection(self.input_variables) - set(self.static_variables)
        if state is None and residual_targets:
            raise ValueError(
                "`state` is required to decode residual targets: " + ", ".join(sorted(residual_targets)) + "."
            )
        outputs: dict[str, Any] = {}
        offset = 0
        for variable, _, levels in self.target_channel_layout:
            values = prediction[:, offset : offset + levels]
            offset += levels
            if state is not None and variable in state and variable not in self.static_variables:
                residual_scale = self._statistic(self.diffs_stddev_by_level, variable)
                last_frame = torch.as_tensor(state[variable], dtype=torch.float32).to(values.device)[:, -1]
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
            if state is not None and variable in self.nan_fill_values and variable in state:
                # The land mask is constant across the conditioning frames, so any frame will do.
                missing = torch.as_tensor(state[variable], dtype=torch.float32).to(values.device).isnan().any(dim=1)
                if missing.ndim == values.ndim - 1:
                    missing = missing[:, None]
                values = torch.where(missing, torch.nan, values)
            values = values[:, 0] if levels == 1 else values
            outputs[variable] = values.numpy() if as_numpy else values
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

        Args:
            state (`Mapping[str, array]`):
                The state the forecast was produced from.
            forecast (`Mapping[str, array]`):
                Physical values from [`~WeatherNext2FeatureExtractor.postprocess`].
            seconds_since_epoch (`np.ndarray` of shape `(batch,)`):
                Valid time of `forecast`, which is the same value that was passed to `__call__` to produce it. The
                appended frame is stamped with it, so advancing the clock before calling this puts the physical
                fields and the clock variables a step out of sync.
        """
        next_state: dict[str, Any] = {}
        forcings = self.compute_forcings(np.asarray(seconds_since_epoch))
        for variable in self.input_variables:
            previous = torch.as_tensor(state[variable], dtype=torch.float32)
            as_numpy = not isinstance(state[variable], torch.Tensor)
            if variable in self.static_variables:
                updated = previous
            elif variable in forcings:
                latest = torch.as_tensor(forcings[variable], dtype=torch.float32).to(previous.device)[:, None]
                updated = torch.cat([previous[:, 1:], latest], dim=1)
            else:
                latest = torch.as_tensor(forecast[variable], dtype=torch.float32).to(previous.device)[:, None]
                updated = torch.cat([previous[:, 1:], latest], dim=1)
            next_state[variable] = updated.numpy() if as_numpy else updated
        return next_state


__all__ = ["WeatherNext2FeatureExtractor"]
