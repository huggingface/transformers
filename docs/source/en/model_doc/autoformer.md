<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-06-24 and added to Hugging Face Transformers on 2023-05-30 and contributed by [elisim](https://huggingface.co/elisim) and [kashif](https://huggingface.co/kashif).*

# Autoformer

[Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://huggingface.co/papers/2106.13008) addresses the challenge of long-term time series forecasting by introducing a novel decomposition architecture. Autoformer integrates an Auto-Correlation mechanism that progressively decomposes trend and seasonal components, enhancing the model's ability to capture intricate temporal patterns. This approach surpasses traditional self-attention methods in both efficiency and accuracy, achieving state-of-the-art results with a 38% relative improvement across six benchmarks in diverse applications including energy, traffic, economics, weather, and disease forecasting.

<hfoptions id="usage">
<hfoption id="AutoformerForPrediction">

```py
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoformerForPrediction

file = hf_hub_download(
    repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly", dtype="auto")
outputs = model.generate(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"],
    future_time_features=batch["future_time_features"],
)

mean_prediction = outputs.sequences.mean(dim=1)
```

</hfoption>
</hfoptions>

## AutoformerConfig

[[autodoc]] AutoformerConfig

## AutoformerModel

[[autodoc]] AutoformerModel
    - forward

## AutoformerForPrediction

[[autodoc]] AutoformerForPrediction
    - forward

