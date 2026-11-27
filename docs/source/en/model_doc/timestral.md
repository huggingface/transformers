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
*This model was published in HF papers on 2026-10-15 and contributed to Hugging Face Transformers on 2026-11-26.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Timestral

## Overview

Timestral is a family of time series foundation models from Mistral AI for probabilistic forecasting, described in
[Timestral](https://huggingface.co/papers/2610.37585). The models forecast univariate and multivariate series
zero-shot, and can condition on covariates known over the context only or over the context and the horizon.

A series is normalized causally, patch by patch, and embedded from its values and its missing-value mask. The
transformer runs over the context patches followed by learned forecast tokens, one per predicted patch. Its layers
attend over time (the patches of a variate, with rotary position embeddings) and, in some layers, over the variates
of a series. A quantile head reads non-crossing quantiles for every predicted time step from the forecast tokens.

This model was contributed by [Abdelaziz03](https://github.com/Abdelaziz03). The original code can be found
[here](https://github.com/mistralai/timestral).

The checkpoints are released under a Modified MIT license. See the model cards on the Hub for its terms.

<!-- TODO before merging: authors and abstract of the paper, and the Hub ids of the converted checkpoints. -->

## Usage example

```python
import torch

from transformers import TimestralModelForPrediction


model = TimestralModelForPrediction.from_pretrained("mistralai/Timestral-209M", device_map="auto")

# (batch_size, num_targets, sequence_length), NaN marks a missing value
past_values = torch.sin(torch.linspace(0, 60, 512))[None, None, :].to(model.device)

with torch.no_grad():
    outputs = model(past_values=past_values, prediction_length=96)

median_forecast = outputs.mean_predictions  # (batch_size, num_targets, prediction_length)
quantile_forecast = outputs.full_predictions  # (batch_size, num_targets, prediction_length, num_quantiles)
```

Covariates are passed next to the targets. `past_covariates` cover the context and `known_covariates` cover the
context and the horizon.

```python
past_covariates = torch.randn(1, 2, 512).to(model.device)
known_covariates = torch.randn(1, 1, 512 + 96).to(model.device)

with torch.no_grad():
    outputs = model(
        past_values=past_values,
        prediction_length=96,
        past_covariates=past_covariates,
        known_covariates=known_covariates,
    )
```

## Notes

- The quantile levels are in `model.config.quantiles`. The quantiles of a time step never cross.
- The targets of a series are forecast jointly. The series of a batch do not interact.
- The model reads at most `config.context_length` time steps and keeps the most recent ones. Shorter contexts are
  left-padded to a multiple of `config.patch_length`.
- One forward pass of the transformer predicts up to `config.block_size` patches. Longer horizons are decoded block by
  block, with the median of a block fed back as context for the next one.
- By default the forecast is averaged with the reflected forecast of the negated inputs (`symmetric_averaging`), and
  the forecast of a series whose observed context is non-negative is clamped at 0 (`clip_nonnegative`). Both can be
  turned off in the call or in the config.
- Categorical covariates must be encoded as numbers first. The original implementation uses a smoothed mean-target
  encoding per series.

## TimestralConfig

[[autodoc]] TimestralConfig

## TimestralModel

[[autodoc]] TimestralModel
    - forward

## TimestralModelForPrediction

[[autodoc]] TimestralModelForPrediction
    - forward
