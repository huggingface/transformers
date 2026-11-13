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

import math

import torch


def latitude_weights(num_latitudes: int, device: torch.device) -> torch.Tensor:
    """Area of each latitude band on an equiangular grid with both poles, normalized to mean 1."""
    if num_latitudes < 2:
        raise ValueError("Latitude weighting requires at least two latitudes, including both poles.")

    latitudes = torch.linspace(-90, 90, num_latitudes, device=device)
    half_spacing = math.pi / (2 * (num_latitudes - 1))
    weights = torch.cos(torch.deg2rad(latitudes)) * math.sin(half_spacing)
    # The polar bands are half as wide as the others.
    weights[0] = weights[-1] = math.sin(half_spacing / 2) ** 2
    return weights / weights.mean()


def channel_weights(config, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per output channel: pressure-level weight, variable weight, and whether missing targets are masked."""
    level_weights, variable_weights, mask_nan = [], [], []
    for variable, _, levels in config.target_channel_layout:
        pressure = config.pressure_levels if variable in config.atmospheric_variables else [1]
        mean_pressure = sum(pressure) / len(pressure)
        level_weights.extend(level / mean_pressure for level in pressure)
        variable_weights.extend([config.loss_weights.get(variable, 1.0) / levels] * levels)
        mask_nan.extend([variable in config.vars_with_nan_targets] * levels)

    return (
        torch.tensor(level_weights, device=device),
        torch.tensor(variable_weights, device=device),
        torch.tensor(mask_nan, device=device, dtype=torch.bool),
    )


def WeatherNext2ForWeatherForecastingLoss(prediction: torch.Tensor, labels: torch.Tensor, config, **kwargs):
    """The original FGN unbiased ensemble CRPS on normalized targets.

    `prediction` flattens the initial-condition and member axes of `labels`, which is shaped
    `(batch_size, num_members, channels, latitudes, longitudes)`. Errors are weighted by latitude band area,
    pressure level and the checkpoint's per-variable `loss_weights`; missing targets of the
    `vars_with_nan_targets` contribute zero to both CRPS terms.
    """
    if config.loss_weights is None:
        raise ValueError("Set `config.loss_weights` from the original checkpoint before computing the loss.")
    if labels.ndim != 5 or prediction.shape != (labels.shape[0] * labels.shape[1], *labels.shape[2:]):
        raise ValueError(
            "`labels` must have shape (batch_size, num_members, channels, latitudes, longitudes), "
            "matching the flattened prediction batch."
        )

    num_members = labels.shape[1]
    if num_members < 2:
        raise ValueError("Unbiased ensemble CRPS requires at least two members per initial condition.")

    prediction = prediction.float().reshape(labels.shape)
    labels = labels.to(device=prediction.device, dtype=torch.float32)
    # Fix the target across members, including residuals from different initial states.
    shift = labels - labels.nanmean(dim=1, keepdim=True)
    targets = labels - shift
    prediction = (prediction - shift).masked_fill(targets.isnan(), float("nan"))

    latitude = latitude_weights(prediction.shape[-2], prediction.device).view(1, 1, 1, -1, 1)
    level, variable, mask_nan = (
        weights.view(1, 1, -1, 1, 1) for weights in channel_weights(config, prediction.device)
    )

    def weighted_mae(predicted, target):
        valid = ~(mask_nan & target.isnan())
        error = (predicted.masked_fill(~valid, 0) - target.masked_fill(~valid, 0)).abs()
        weighted = error * latitude * level * variable
        return weighted.sum(dim=2).mean(dim=(1, 2, 3))

    absolute_error = prediction.new_zeros(prediction.shape[0])
    absolute_difference = prediction.new_zeros(prediction.shape[0])
    for i in range(num_members):
        absolute_error = absolute_error + weighted_mae(prediction[:, i : i + 1], targets)
        for j in range(i):
            absolute_difference = absolute_difference + weighted_mae(
                prediction[:, j : j + 1], prediction[:, i : i + 1]
            )

    crps = absolute_error / num_members - absolute_difference / (num_members * (num_members - 1))
    return crps.mean()
