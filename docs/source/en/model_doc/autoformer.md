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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Autoformer

[Autoformer](https://huggingface.co/papers/2106.13008) is a model designed for long-term forecasting. It modifies the Transformer architecture with decomposition blocks that breaks down long-term trends, and it also adds an Auto-Correlation mechanism in place of self-attention to discover repeating patterns. The model combines these two designs to make better predictions.

You can find all the original Autoformer checkpoints on the [Hub](https://huggingface.co/models?search=autoformer).

> [!TIP]
> Click on the Autoformer model cards in the right sidebar to explore more examples and use cases like weather or energy time series forecasting.

The example below demonstrates how to make a prediction with [`AutoformerForPrediction`].

<hfoptions id="usage">

<hfoption id="AutoformerForPrediction">

```py
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoformerForPrediction, AutoformerConfig

file = hf_hub_download(
    repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly")

outputs = model.generate(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"],
    future_time_features=batch["future_time_features"],
)

mean_prediction = outputs.sequences.mean(dim=1)
print(f"Mean prediction: {mean_prediction}")
```
</hfoption> 

</hfoptions>

Quantization
Since Autoformer is lightweight and not extremely parameter-heavy like LLMs, quantization isn't typically needed. However, if you're experimenting with edge deployment or want to reduce size even more, standard PyTorch quantization techniques (like dynamic quantization for linear layers) can be applied.

```py
from torch.ao.quantization import quantize_dynamic
from transformers import AutoformerForPrediction

model = AutoformerForPrediction.from_pretrained("elisim/autoformer-energy")
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

Attention Mask Visualization
Autoformer currently does not use standard attention mechanisms, so it isn't compatible with AttentionMaskVisualizer. Instead, it uses an Auto-Correlation mechanism, which doesn't operate on per-token attention masks.

## Notes

- Read the [Yes, Transformers are Effective for Time Series Forecasting (+ Autoformer)](https://huggingface.co/blog/autoformer) blog post for more details about using Autoformer to make predictions.

## AutoformerConfig

[[autodoc]] AutoformerConfig

## AutoformerModel

[[autodoc]] AutoformerModel
    - forward

## AutoformerForPrediction

[[autodoc]] AutoformerForPrediction
    - forward