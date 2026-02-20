<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-10-14 and added to Hugging Face Transformers on 2026-02-18.*

# TimesFM 2.5

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

TimesFM 2.5 is available in Transformers format as:
[`google/timesfm-2.5-200m-transformers`](https://huggingface.co/google/timesfm-2.5-200m-transformers).
This Transformers implementation ports the official 2.5 architecture with rotary attention, QK normalization,
per-dimension attention scaling, and continuous quantile prediction.

The upstream model card reports an architecture update on October 2, 2025 (fused QKV for speed, unchanged results),
and points to the TimesFM paper [2310.10688](https://huggingface.co/papers/2310.10688).

The abstract from the paper is the following:

*Motivated by recent advances in large language models for Natural Language Processing (NLP), we design a time-series foundation model for forecasting whose out-of-the-box zero-shot performance on a variety of public datasets comes close to the accuracy of state-of-the-art supervised forecasting models for each individual dataset. Our model is based on pretraining a decoder style attention model with input patching, using a large time-series corpus comprising both real-world and synthetic datasets. Experiments on a diverse set of previously unseen forecasting datasets suggests that the model can yield accurate zero-shot forecasts across different domains, forecasting horizons and temporal granularities.*

The model was contributed by [kashif](https://huggingface.co/kashif).

## Usage example

```python
import numpy as np
import torch
from transformers import Timesfm2P5ModelForPrediction


model = Timesfm2P5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers",
    attn_implementation="sdpa",
    dtype=torch.float32,
    device_map="auto",
)

forecast_input = [
    np.sin(np.linspace(0, 20, 100)),
    np.sin(np.linspace(0, 20, 200)),
    np.sin(np.linspace(0, 20, 400)),
]
forecast_input_tensor = [torch.tensor(ts, dtype=torch.float32, device=model.device) for ts in forecast_input]

with torch.no_grad():
    outputs = model(past_values=forecast_input_tensor, return_dict=True)
    point_forecast = outputs.mean_predictions
    quantile_forecast = outputs.full_predictions
```

## Timesfm2P5Config

[[autodoc]] Timesfm2P5Config

## Timesfm2P5Model

[[autodoc]] Timesfm2P5Model
    - forward

## Timesfm2P5ModelForPrediction

[[autodoc]] Timesfm2P5ModelForPrediction
    - forward
