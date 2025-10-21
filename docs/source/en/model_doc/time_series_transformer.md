<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2022-09-30 and contributed by [kashif](https://huggingface.co/kashif).*

# Time Series Transformer

[Time Series Transformer](https://huggingface.co/blog/time-series-transformers) is a a vanilla Encoder-Decoder Transformer for univariate probabilistic time series forecasting. The model leverages an autoregressive setup, where the encoder processes a context window of past data and the causal-masked decoder predicts future values using ancestral sampling. Transformers handle long time series efficiently by training on fixed context and prediction windows and can naturally incorporate missing values via masking, avoiding imputation. Limitations include quadratic compute and memory costs, restricting window sizes, and a higher risk of overfitting due to the model's capacity.

<hfoptions id="usage">
<hfoption id="TimeSeriesTransformerForPrediction">

```py
import torch
from huggingface_hub import hf_hub_download
from transformers import TimeSeriesTransformerForPrediction

file = hf_hub_download(
    repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = TimeSeriesTransformerForPrediction.from_pretrained("huggingface/time-series-transformer-tourism-monthly", dtype="auto")

outputs = model.generate(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"],
    static_real_features=batch["static_real_features"],
    future_time_features=batch["future_time_features"],
)

mean_prediction = outputs.sequences.mean(dim=1)
```

</hfoption>
</hfoptions>

## Usage tips

- [`TimeSeriesTransformerModel`] is the raw Transformer without any head. [`TimeSeriesTransformerForPrediction`] adds a distribution head for time-series forecasting. This is a probabilistic forecasting model, not a point forecasting model. The model learns a distribution from which you sample rather than outputting values directly.
- [`TimeSeriesTransformerForPrediction`] has two blocks: an encoder that takes `context_length` time series values as input (`past_values`) and a decoder that predicts `prediction_length` time series values into the future (`future_values`). During training, provide pairs of `past_values` and `future_values` to the model.
- Provide additional features alongside the raw `past_values` and `future_values`:

  - `past_time_features`: Temporal features added to `past_values`. These serve as positional encodings for the Transformer encoder. Examples include "day of the month" and "month of the year" as scalar values stacked into a vector. For example, if a time-series value was obtained on August 11th, you'd have `[11, 8]` as the time feature vector (11 for day of month, 8 for month of year).
  - `future_time_features`: Temporal features added to `future_values`. These serve as positional encodings for the Transformer decoder. Examples include "day of the month" and "month of the year" as scalar values stacked into a vector. For example, if a time-series value was obtained on August 11th, you'd have `[11, 8]` as the time feature vector (11 for day of month, 8 for month of year).
  - `static_categorical_features`: Categorical features that remain constant over time (same value for all `past_values` and `future_values`). Examples include store ID or region ID that identifies a given time-series. These features must be known for ALL data points, including future ones.
  - `static_real_features`: Real-valued features that remain constant over time (same value for all `past_values` and `future_values`). Examples include image representations of products (like ResNet embeddings of product pictures). These features must be known for ALL data points, including future ones.

- The model trains using teacher-forcing, similar to how Transformers train for machine translation. During training, shift `future_values` one position to the right as input to the decoder, prepended by the last value of `past_values`. At each time step, the model predicts the next target. The training setup resembles a GPT model for language, except there's no `decoder_start_token_id` (use the last value of the context as initial input for the decoder).
- At inference time, give the final value of `past_values` as input to the decoder. Sample from the model to make a prediction at the next time step, then feed that prediction back to the decoder to make the next prediction (autoregressive generation).

## TimeSeriesTransformerConfig

[[autodoc]] TimeSeriesTransformerConfig

## TimeSeriesTransformerModel

[[autodoc]] TimeSeriesTransformerModel
    - forward

## TimeSeriesTransformerForPrediction

[[autodoc]] TimeSeriesTransformerForPrediction
    - forward
