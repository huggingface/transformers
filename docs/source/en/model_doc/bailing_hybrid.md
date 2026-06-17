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

# BailingHybrid

## Overview

The BailingHybrid model (Ring-2.5-1T / Ling-2.5-1T) was proposed by [InclusionAI](https://huggingface.co/inclusionAI). It is a trillion-parameter model based on a hybrid linear attention architecture, combining Multi-head Latent Attention (MLA), Lightning Linear Attention, and Mixture of Experts (MoE).

Key architectural features:
- **Hybrid Attention**: Uses a 1:7 ratio of MLA to Lightning Linear Attention layers, achieving near-linear computational complexity
- **Multi-head Latent Attention (MLA)**: Similar to DeepSeek-V3, with compressed KV cache via LoRA projections
- **Lightning Linear Attention**: Based on SimpleGLA (Simple Gated Linear Attention) from the flash-linear-attention library
- **Mixture of Experts**: 256 routed experts with 8 active per token, plus shared experts

### Usage tips

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "inclusionAI/Ring-2.5-1T",
    device_map="auto",
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ring-2.5-1T")

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

For optimal performance with the linear attention layers, install the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) library. Without it, the model falls back to a pure PyTorch implementation.

## BailingHybridConfig

[[autodoc]] BailingHybridConfig

## BailingHybridModel

[[autodoc]] BailingHybridModel
    - forward

## BailingHybridForCausalLM

[[autodoc]] BailingHybridForCausalLM
    - forward

## BailingHybridForSequenceClassification

[[autodoc]] BailingHybridForSequenceClassification
    - forward

## BailingHybridForTokenClassification

[[autodoc]] BailingHybridForTokenClassification
    - forward
