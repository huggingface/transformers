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

The Cisco Time Series Model (CTSM) was proposed in [Cisco Time Series Model Technical Report](https://huggingface.co/papers/2511.19841) by Liang Gou, Archit Khare, Praneet Pabolu, Prachi Patel, Joseph Ross, Hercy Shen, Yuhan (Ellen) Song, Jingze Sun, Kristal Curtis, Vedant Dharnidharka, Abhinav Mathur and Hao Yang.

CTSM is a decoder-only univariate zero-shot forecasting foundation model. Its central idea is a **multi-resolution context**: instead of consuming a single-scale history, each forecast conditions on two aligned streams — a coarse low-frequency stream (e.g. 512 hourly points) and a fine high-frequency stream (e.g. 512 minutely points), with the resolution ratio fixed to 60. A learnable **special token** separates the two streams and learned **resolution embeddings** are added to the token stream to distinguish them. The coarse stream lets the model see week-over-week structure without giving up fine-grained recent detail; as the paper puts it, "more complex multiresolution architectures would require a context length of 30,720 (30 times as long as ours) to cover the same time range."

The abstract from the paper is the following:

*We introduce the Cisco Time Series Model, a univariate zero-shot forecaster. This time series foundation model is the result of a general architectural innovation to a time series model enabling it to accept multiresolution input, applied to a popular decoder-only time series model (TimesFM). The resulting multiresolution decoder-only model is trained on over 300B unique data points, with more than half coming from the observability domain. Quantitative and qualitative evaluations demonstrate that the resulting model achieves superior performance on observability datasets while retaining very similar performance on a standard general-purpose forecasting benchmark (GIFT-Eval), and suggest that the multiresolution structure enables the model to make more accurate predictions on long context input.*

### Architecture

The backbone follows TimesFM 2.0: patching (patch length 32) + a residual-block input tokenizer + decoder-only transformer layers with per-dimension learnable query scaling + a residual-block horizon head. CTSM adds, on top:

- A **special token** inserted between the coarse and fine patch streams, so the input is `[coarse₁, …, coarse₁₆, SPECIAL, fine₁, …, fine₁₆]`.
- **Resolution embeddings** (3-way: coarse / special / fine) added to each token before the transformer stack.
- **Stream-level normalization**: each stream is standardized independently over its non-padded context, and the fine-stream statistics are used to rescale the forecast.
- A **frequency embedding** inherited from TimesFM, added to every token.

The 250M **CTSM 1.0** release checkpoint additionally introduces (over the 500M `1.0-preview` described in the paper):

- **Rotary position embeddings (RoPE)** applied to query/key inside attention.
- **Bidirectional attention over the coarse block** — tokens in the coarse segment attend both ways within that segment, while the fine segment remains causal.
- **15-quantile prediction** (levels 0.01–0.99) instead of 9.
- **Short-context training** (1/3 of training samples drawn with `|fine| ∈ [10, 511]`) for better robustness when less history is available.
- Trained from scratch (not continued pre-training from TimesFM 2.0) on ~2× more internal observability data.

### Inference

For `horizon_len > config.horizon_length`, [`CtsmModelForPrediction`] runs an autoregressive multi-resolution decode loop, using a [`DynamicCache`] by default (opt out with `use_cache=False`). Each step feeds only the newly-appended fine patches through the stack and attends to cached K/V for every earlier position. Stream-normalization statistics are frozen to their step-1 values so that cached K/V remains valid; the coarse block is pinned and the cache is rebuilt if the concatenated sequence would outgrow `max_position_embeddings`.

The checkpoint can be found at [`cisco-ai/cisco-time-series-model-1.0`](https://huggingface.co/cisco-ai/cisco-time-series-model-1.0). The original inference code is at [github.com/splunk/cisco-time-series-model](https://github.com/splunk/cisco-time-series-model).

This model was contributed by [kashif](https://huggingface.co/kashif).

## Usage

Pass a list of fine-resolution time series (e.g. minute-level); the coarse stream is built automatically by mean-aggregating consecutive blocks of `config.agg_factor` points.

```python
import numpy as np
import torch
from transformers import CtsmModelForPrediction


model = CtsmModelForPrediction.from_pretrained("cisco-ai/cisco-time-series-model-1.0", device_map="auto")

# ~8.5 hours of 1-minute data; the model will build a 512-hour coarse context by aggregation.
series = np.sin(np.linspace(0, 200, 512 * 60)).astype(np.float32)
past_values = [torch.tensor(series, device=model.device)]

with torch.no_grad():
    outputs = model(past_values=past_values, horizon_len=128)

point_forecast = outputs.mean_predictions       # (batch, horizon_len)
quantile_forecast = outputs.full_predictions    # (batch, horizon_len, 1 + num_quantiles)
```

If you already have a coarse stream (e.g. pre-computed 1-hour roll-ups that go further back than you have 1-minute data for), pass `(coarse, fine)` pairs directly:

```python
coarse = torch.tensor(hourly_series, dtype=torch.float32)    # up to 512 points
fine = torch.tensor(minutely_series, dtype=torch.float32)    # up to 512 points
outputs = model(past_values=[(coarse, fine)], horizon_len=128)
```

For `horizon_len > 128`, the model decodes autoregressively and extends the output accordingly.

## CtsmConfig

[[autodoc]] CtsmConfig

## CtsmModel

[[autodoc]] CtsmModel
    - forward

## CtsmModelForPrediction

[[autodoc]] CtsmModelForPrediction
    - forward

## Citation

```bibtex
@misc{gou2025ciscotimeseriesmodel,
      title={Cisco Time Series Model Technical Report},
      author={Liang Gou and Archit Khare and Praneet Pabolu and Prachi Patel and Joseph Ross and Hercy Shen and Yuhan Song and Jingze Sun and Kristal Curtis and Vedant Dharnidharka and Abhinav Mathur and Hao Yang},
      year={2025},
      eprint={2511.19841},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.19841}
}
```
