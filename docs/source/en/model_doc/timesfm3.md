<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-26.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# TimesFM 3

## Overview

TimesFM 3 is a time series foundation model from Google Research, introduced in the blog post [TimesFM-3: A zero-shot foundation model for multivariate forecasting](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/). It follows [TimesFM 2.5](./timesfm2_5) and forecasts several related series jointly, with or without covariates, without any training on the task.

The model reads a grid of patches, one row per series and one column per patch of 32 time steps. Each layer applies a causal attention over time, within each series, then an attention over the series, within each column. This is the attention of TimesFM 2.5, with rotary positions over time, normalized queries and keys and a learned scale per dimension. The other differences with TimesFM 2.5 are the following.

- **Covariates.** A covariate is one more series in the grid. It is either observed over the context only, or known in advance over the horizon too, like a calendar of promotions. Each patch of a series known in advance also carries its values over the next 64 time steps.
- **One forward pass.** The patches of the horizon are appended to the context as masked patches, so a forecast of any length is read from a single pass. Each patch predicts the next 64 time steps, and the predictions of consecutive patches are blended over the steps they share.
- **Normalization.** Each patch is normalized with the mean and the standard deviation of its series up to that patch. Over the horizon, these statistics are extended with the predicted medians. A linear trend is removed from a series, and added back to its forecast, when removing it at least halves the standard deviation of the series.
- **Quantiles.** The model predicts nine quantiles, from 0.1 to 0.9. The point forecast is the median.

This model was contributed by [Abdelaziz03](https://github.com/Abdelaziz03). The original code can be found [here](https://github.com/google-research/timesfm).

> [!NOTE]
> The code is released under the Apache 2.0 license, but the checkpoint [`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch) is released under the `timesfm-non-commercial-license-v1.0` license, which restricts it to non-commercial and non-production use. See the model card on the Hub for its terms.

<!-- TODO: point to the converted checkpoint once it is on the Hub. -->

## Usage example

Pass the target series as a tensor of shape `(batch_size, num_targets, sequence_length)`. The targets of a batch element are forecast jointly.

```python
import torch

from transformers import TimesFm3ModelForPrediction


model = TimesFm3ModelForPrediction.from_pretrained("google/timesfm-3.0-pytorch", device_map="auto")

steps = torch.arange(512, dtype=torch.float32)
past_values = torch.stack([torch.sin(steps / 10), torch.cos(steps / 10)])[None].to(model.device)

with torch.no_grad():
    outputs = model(past_values=past_values, prediction_length=96)

point_forecast = outputs.mean_predictions  # (1, 2, 96)
quantile_forecast = outputs.full_predictions  # (1, 2, 96, 9)
```

Covariates are passed next to the targets. `past_covariates` are observed over the context only, and `known_covariates` over the context and the horizon.

```python
prediction_length = 96
past_covariates = torch.randn(1, 1, 512, device=model.device)
known_covariates = torch.randn(1, 2, 512 + prediction_length, device=model.device)

with torch.no_grad():
    outputs = model(
        past_values=past_values,
        prediction_length=prediction_length,
        past_covariates=past_covariates,
        known_covariates=known_covariates,
    )
```

## Usage tips

- Missing values are NaN. The time steps before the first observation of a batch element are ignored, and the other missing values are interpolated linearly, as in the original implementation.
- The context is cut to its last `config.context_length` time steps, 15,360 by default.
- `force_flip_invariance=True` also forecasts the negated inputs and averages the two forecasts, and `truncate_negative=True` clamps to non-negative values the forecasts of the series that have no negative value. The original implementation uses both in its benchmarks.
- Without known covariates, the horizon is decoded in multiples of `config.horizon_length` and cut to `prediction_length`.

## TimesFm3Config

[[autodoc]] TimesFm3Config

## TimesFm3Model

[[autodoc]] TimesFm3Model
    - forward

## TimesFm3ModelForPrediction

[[autodoc]] TimesFm3ModelForPrediction
    - forward
