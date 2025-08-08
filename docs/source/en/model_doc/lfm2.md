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

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# LFM2

## Overview

[LFM2](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models) represents a new generation of Liquid Foundation Models developed by [Liquid AI](https://liquid.ai/), specifically designed for edge AI and on-device deployment. 

The models are available in three sizes (350M, 700M, and 1.2B parameters) and are engineered to run efficiently on CPU, GPU, and NPU hardware, making them particularly well-suited for applications requiring low latency, offline operation, and privacy.

## Architecture

The architecture consists of 16 blocks total: 10 double-gated short-range convolution blocks and 6 blocks of grouped query attention. This design stems from the concept of dynamical systems, where linear operations are modulated by input-dependent gates, allowing for "liquid" dynamics that can adapt in real-time. The short convolutions are particularly optimized for embedded SoC CPUs, making them ideal for devices that require fast, local inference without relying on cloud connectivity.

The key architectural innovation of LFM2 lies in its systematic approach to balancing quality, latency, and memory efficiency through our STAR neural architecture search engine. Using STAR, Liquid AI optimized the models for real-world performance on embedded hardware, measuring actual peak memory usage and inference speed on Qualcomm Snapdragon processors. This results in models that achieve 2x faster decode and prefill performance compared to similar-sized models, while maintaining superior benchmark performance across knowledge, mathematics, instruction following, and multilingual tasks.

## Example

The following example shows how to generate an answer using the `AutoModelForCausalLM` class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_id = "LiquidAI/LFM2-1.2B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Generate answer
prompt = "What is C. elegans?"
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
)

output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.05,
    max_new_tokens=512,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))
```

## Lfm2Config

[[autodoc]] Lfm2Config

## Lfm2Model

[[autodoc]] Lfm2Model
    - forward

## Lfm2ForCausalLM

[[autodoc]] Lfm2ForCausalLM
    - forward