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

## TimeSeriesTransformerConfig

[[autodoc]] TimeSeriesTransformerConfig

## TimeSeriesTransformerModel

[[autodoc]] TimeSeriesTransformerModel
    - forward

## TimeSeriesTransformerForPrediction

[[autodoc]] TimeSeriesTransformerForPrediction
    - forward
