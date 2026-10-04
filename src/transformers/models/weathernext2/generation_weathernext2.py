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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ...modeling_outputs import ModelOutput


if TYPE_CHECKING:
    from .feature_extraction_weathernext2 import WeatherNext2FeatureExtractor


@dataclass
class WeatherNext2GenerationOutput(ModelOutput):
    """Output of [`WeatherNext2GenerationMixin.generate`].

    Args:
        forecasts (`list[dict[str, torch.Tensor]]`):
            One postprocessed forecast in physical units per generated time step. With `num_members > 1` the batch
            axis holds every member of every initial condition: member `m` of initial condition `i` is row
            `i * num_members + m`.
        state (`dict[str, torch.Tensor]`):
            Conditioning state after the final forecast has been appended.
        valid_time (`torch.Tensor`):
            Valid time, as Unix seconds, of the next forecast that would be generated.
    """

    forecasts: list[dict[str, torch.Tensor]] | None = None
    state: dict[str, torch.Tensor] | None = None
    valid_time: torch.Tensor | None = None


class WeatherNext2GenerationMixin:
    """Autoregressive generation methods for WeatherNext 2."""

    @torch.no_grad()
    def generate(
        self,
        state: Mapping[str, Any],
        feature_extractor: "WeatherNext2FeatureExtractor",
        seconds_since_epoch: torch.Tensor,
        num_steps: int,
        num_members: int | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        **model_kwargs,
    ) -> WeatherNext2GenerationOutput:
        """Generates an autoregressive weather forecast.

        Args:
            state (`Mapping[str, array]`):
                Physical conditioning state in the format accepted by
                [`WeatherNext2FeatureExtractor.__call__`].
            feature_extractor (`WeatherNext2FeatureExtractor`):
                Feature extractor used to normalize inputs, decode predictions and advance the state.
            seconds_since_epoch (`torch.Tensor` of shape `(batch_size,)`):
                Valid time of the first generated forecast, as Unix seconds.
            num_steps (`int`):
                Number of autoregressive time steps to generate.
            num_members (`int`, *optional*):
                Number of ensemble members drawn from each initial condition. Each one is repeated `num_members`
                times along the batch axis, and every member gets its own noise. Defaults to `config.num_members`.
            noise (`torch.Tensor` of shape `(num_steps, batch_size * num_members, noise_channels)`, *optional*):
                Explicit noise vector for every step and ensemble member. When omitted, a fresh vector is sampled
                at every step.
            generator (`torch.Generator`, *optional*):
                Generator used to sample noise when `noise` is omitted.
            model_kwargs (`dict`, *optional*):
                Additional arguments forwarded to the model at every step.

        Returns:
            [`WeatherNext2GenerationOutput`] containing each forecast in physical units, the final conditioning
            state, and the valid time of the next forecast.
        """
        if num_steps < 1:
            raise ValueError(f"`num_steps` must be at least 1, got {num_steps}.")

        num_members = num_members if num_members is not None else self.config.num_members
        if num_members < 1:
            raise ValueError(f"`num_members` must be at least 1, got {num_members}.")
        valid_time = torch.as_tensor(seconds_since_epoch, dtype=torch.int64)
        if valid_time.ndim != 1:
            raise ValueError(f"`seconds_since_epoch` must have shape (batch_size,), got {tuple(valid_time.shape)}.")

        # Every initial condition becomes `num_members` rows; static fields have no batch axis to repeat.
        state = {
            name: values
            if name in feature_extractor.static_variables
            else torch.as_tensor(values).repeat_interleave(num_members, dim=0)
            for name, values in state.items()
        }
        valid_time = valid_time.repeat_interleave(num_members)
        batch_size = valid_time.shape[0]
        if noise is not None:
            expected_shape = (num_steps, batch_size, self.config.noise_channels)
            if tuple(noise.shape) != expected_shape:
                raise ValueError(f"`noise` has shape {tuple(noise.shape)}, expected {expected_shape}.")

        current_state = dict(state)
        forecasts = []
        step_seconds = feature_extractor.time_step_hours * 3600
        for step in range(num_steps):
            inputs = feature_extractor(current_state, seconds_since_epoch=valid_time, device=self.device)
            outputs = self(
                **inputs,
                noise=None if noise is None else noise[step].to(self.device),
                generator=generator,
                **model_kwargs,
            )
            forecast = feature_extractor.postprocess(outputs.prediction, current_state)
            forecasts.append(forecast)
            current_state = feature_extractor.advance_state(current_state, forecast, valid_time)
            valid_time = valid_time + step_seconds

        return WeatherNext2GenerationOutput(forecasts=forecasts, state=current_state, valid_time=valid_time)


__all__ = ["WeatherNext2GenerationMixin", "WeatherNext2GenerationOutput"]
