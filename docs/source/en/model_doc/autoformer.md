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

[Autoformer](https://huggingface.co/papers/2106.13008) is a time series forecasting model that rethinks Transformers with a twist — instead of just using attention, it brings in a **decomposition-based architecture** and a new **Auto-Correlation mechanism** to capture complex patterns in time series data.

What makes Autoformer unique? It doesn't treat trend and seasonal patterns as noise — it learns to **decompose** them, improving both performance and interpretability. Plus, it's super efficient for **long-term forecasting**, avoiding the bottlenecks of traditional Transformer-based approaches.

It’s especially great for scenarios like energy consumption, traffic flow, economic indicators, weather patterns, and disease spread forecasting.

You can explore the model on the [Autoformer Hub Page](https://huggingface.co/elisim/autoformer-energy) or other community-hosted variants.

> [!TIP]
> Click on the Autoformer model cards in the right sidebar to explore more examples and use cases like weather or energy time series forecasting.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`]

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

forecast = pipeline(task="time-series-forecasting", model="elisim/autoformer-energy")
output = forecast("energy.csv")  # this assumes a supported dataset format
```
</hfoption> 

<hfoption id="AutoModel">

```py
import torch
from transformers import AutoformerForPrediction, AutoformerConfig

model = AutoformerForPrediction.from_pretrained("elisim/autoformer-energy")
config = AutoformerConfig.from_pretrained("elisim/autoformer-energy")

# Autoformer expects inputs like encoder_input: [batch_size, context_len, input_dim]
# Depending on how the model is implemented, check for the exact expected input name

dummy_input = torch.rand(1, config.seq_len, config.input_size)

# Wrap the tensor in a dictionary, assuming input is named "past_values" or similar
output = model(past_values=dummy_input)  # Replace with correct input name if different

print(output)
```
</hfoption> 

<hfoption id="transformers-cli">

```bash
transformers-cli env  # transformers-cli doesn't currently support time series inputs
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

- Autoformer was proposed in the paper [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008) by Haixu Wu et al.
- The model was contributed by [@elisim](https://huggingface.co/elisim) and [@kashif](https://huggingface.co/kashif).
- The official implementation is available on [GitHub](https://github.com/thuml/Autoformer).
- Check out Hugging Face’s blog post on this model: [Yes, Transformers are Effective for Time Series Forecasting (+ Autoformer)](https://huggingface.co/blog/autoformer).

## AutoformerConfig

[[autodoc]] AutoformerConfig

## AutoformerModel

[[autodoc]] AutoformerModel
- forward

## AutoformerForPrediction

[[autodoc]] AutoformerForPrediction
- forward