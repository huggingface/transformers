<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

*This model was released on 2023-10-14 and added to Hugging Face Transformers on 2025-04-16 and contributed by [kashif](https://huggingface.co/kashif).*

# TimesFM

[TimesFM](https://huggingface.co/papers/2310.10688) is a pretrained time-series foundation model designed for forecasting. It leverages a patched-decoder style attention mechanism and is trained on a large time-series corpus. The model demonstrates strong zero-shot performance across various datasets, matching or nearly matching the accuracy of specialized supervised forecasting models. It handles different forecasting history lengths, prediction lengths, and temporal granularities effectively.

<hfoptions id="usage">
<hfoption id="TimesFmModelForPrediction">

```py
import torch
from transformers import TimesFmModelForPrediction

model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch", dtype="auto")

forecast_input = [torch.linspace(0, 20, 100).sin(), torch.linspace(0, 20, 200).sin(), torch.linspace(0, 20, 400).sin()]
frequency_input = torch.tensor([0, 1, 2], dtype=torch.long)

with torch.no_grad():
    outputs = model(past_values=forecast_input, freq=frequency_input, return_dict=True)
    point_forecast_conv = outputs.mean_predictions
    quantile_forecast_conv = outputs.full_predictions
```

</hfoption>
</hfoptions>

## TimesFmConfig

[[autodoc]] TimesFmConfig

## TimesFmModel

[[autodoc]] TimesFmModel
    - forward

## TimesFmModelForPrediction

[[autodoc]] TimesFmModelForPrediction
    - forward

