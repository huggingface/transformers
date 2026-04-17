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
*This model was released on 2025-11-25 and added to Hugging Face Transformers on 2026-04-17.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# CTSM

## Overview

The Cisco Time Series Model (CTSM) 1.0 is a 250M-parameter decoder-only foundation model for univariate zero-shot
forecasting, proposed in [Cisco Time Series Model Technical Report](https://huggingface.co/papers/2511.19841) by
Liang Gou et al. It is architecturally inspired by [TimesFM 2.0](https://huggingface.co/google/timesfm-2.0-500m-pytorch)
and adds a multi-resolution context (a coarse stream aggregated by a configurable `agg_factor`, a learned special
token, and a fine stream), rotary position embeddings, bidirectional attention over the coarse-resolution block,
15-quantile prediction, and per-resolution learned embeddings.

The checkpoint can be found at [`cisco-ai/cisco-time-series-model-1.0`](https://huggingface.co/cisco-ai/cisco-time-series-model-1.0).

## Usage example

```python
import numpy as np
import torch
from transformers import CtsmModelForPrediction


model = CtsmModelForPrediction.from_pretrained("cisco-ai/cisco-time-series-model-1.0", device_map="auto")

# A fine-resolution (e.g. minute-level) time series. The coarse stream is built automatically
# by mean-aggregating consecutive blocks of `config.agg_factor` points.
series = np.sin(np.linspace(0, 200, 512 * 60)).astype(np.float32)
past_values = [torch.tensor(series, device=model.device)]

with torch.no_grad():
    outputs = model(past_values=past_values, horizon_len=128)

point_forecast = outputs.mean_predictions  # (batch, horizon_len)
quantile_forecast = outputs.full_predictions  # (batch, horizon_len, 1 + num_quantiles)
```

You can also pass `(coarse, fine)` pairs directly if you already have the coarse stream:

```python
coarse = torch.tensor(coarse_series, dtype=torch.float32)
fine = torch.tensor(fine_series, dtype=torch.float32)
outputs = model(past_values=[(coarse, fine)], horizon_len=128)
```

## CtsmConfig

[[autodoc]] CtsmConfig

## CtsmModel

[[autodoc]] CtsmModel
    - forward

## CtsmModelForPrediction

[[autodoc]] CtsmModelForPrediction
    - forward
