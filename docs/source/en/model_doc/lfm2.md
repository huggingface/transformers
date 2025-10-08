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
*This model was released on 2025-07-10 and added to Hugging Face Transformers on 2025-07-10.*

# LFM2

## Overview

[LFM2](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models) models are ultra-efficient foundation models optimized for on-device use, offering up to 2x faster CPU decoding than Qwen3 and 3x faster training efficiency than the prior generation. They use a new hybrid architecture with multiplicative gates and short convolutions across 16 blocks, achieving strong benchmark performance in knowledge, math, multilingual tasks, and instruction following. LFM2 comes in 0.35B, 0.7B, and 1.2B parameter sizes and consistently outperforms larger peers like Gemma 3 and Llama 3.2 in its class. Designed for phones, laptops, vehicles, and edge devices, these models balance speed, memory efficiency, and privacy for real-time, local AI deployment

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="LiquidAI/LFM2-1.2B", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-1.2B")
model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2-1.2B", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Lfm2Config

[[autodoc]] Lfm2Config

## Lfm2Model

[[autodoc]] Lfm2Model
    - forward

## Lfm2ForCausalLM

[[autodoc]] Lfm2ForCausalLM
    - forward