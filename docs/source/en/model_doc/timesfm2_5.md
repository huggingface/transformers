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
*This model was released on 2025-09-15 and added to Hugging Face Transformers on 2026-02-26.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# TimesFM 2.5

## Overview

TimesFM 2.5 (Time Series Foundation Model) is a pretrained time-series foundation model proposed in [A decoder-only foundation model for time-series forecasting](https://huggingface.co/papers/2310.10688) by Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. It builds on the original TimesFM architecture with rotary attention, QK normalization, per-dimension attention scaling, and continuous quantile prediction.

The abstract from the paper is the following:

*Motivated by recent advances in large language models for Natural Language Processing (NLP), we design a time-series foundation model for forecasting whose out-of-the-box zero-shot performance on a variety of public datasets comes close to the accuracy of state-of-the-art supervised forecasting models for each individual dataset. Our model is based on pretraining a decoder style attention model with input patching, using a large time-series corpus comprising both real-world and synthetic datasets. Experiments on a diverse set of previously unseen forecasting datasets suggests that the model can yield accurate zero-shot forecasts across different domains, forecasting horizons and temporal granularities.*

This model was contributed by [kashif](https://huggingface.co/kashif). The original code can be found [here](https://github.com/google-research/timesfm).

You can find the checkpoint at [`google/timesfm-2.5-200m-transformers`](https://huggingface.co/google/timesfm-2.5-200m-transformers).

## Usage example

```python
import numpy as np
import torch
from transformers import TimesFm2_5ModelForPrediction


model = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers",
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

## TimesFm2_5Config

[[autodoc]] TimesFm2_5Config

## TimesFm2_5Model

[[autodoc]] TimesFm2_5Model
    - forward

## TimesFm2_5ModelForPrediction

[[autodoc]] TimesFm2_5ModelForPrediction
    - forward
