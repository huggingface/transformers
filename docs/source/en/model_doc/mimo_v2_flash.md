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
*This model was released on 2025-12-17 and added to Hugging Face Transformers on 2026-04-30.*

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# MiMo-V2-Flash

## Overview

**MiMo-V2-Flash** is a Mixture-of-Experts (MoE) language model developed by the Xiaomi MiMo team. Designed to establish a new balance between long-context modeling capabilities and inference efficiency, the model is built for strong performance in complex reasoning and agentic tasks. Trained on 27T tokens with native 32k sequence lengths, MiMo-V2-Flash seamlessly supports an extended **256K context window** while significantly reducing KV-cache storage compared to standard global attention models.

### Key Features

- **Hybrid Attention Architecture:** Interleaves Sliding Window Attention (SWA) and Global Attention (GA) at a 5:1 ratio, using an aggressive 128-token window. This approach reduces KV-cache storage by nearly 6x while utilizing a learnable attention sink bias to preserve excellent performance on long contexts.
- **Agentic Capabilities:** Enhanced through Multi-Teacher On-Policy Distillation (MOPD) and large-scale agentic RL during post-training, the model demonstrates superior tool-use capabilities and exceptional performance on benchmarks like SWE-Bench.
- **Inference Efficiency:** Pre-trained using FP8 mixed precision, making it highly optimized for practical deployments and modern accelerators.


For more details, please refer to the [technical
report](https://github.com/XiaomiMiMo/MiMo-V2-Flash/blob/main/paper.pdf), and the [official
repository](https://github.com/XiaomiMiMo/MiMo-V2-Flash).  
This model was contributed by [casinca](https://huggingface.co/casinca).

## Usage examples

### Text generation

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="XiaomiMiMo/MiMo-V2-Flash",
)
pipe("Explain why sparse MoE models can be efficient at inference.")
```

</hfoption>
<hfoption id="AutoModelForCausalLM">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("XiaomiMiMo/MiMo-V2-Flash")
model = AutoModelForCausalLM.from_pretrained(
    "XiaomiMiMo/MiMo-V2-Flash",
    device_map="auto",
)
input_ids = tokenizer("Explain why sparse MoE models can be efficient at inference.", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

### Chat template generation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "XiaomiMiMo/MiMo-V2-Flash"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are MiMo, a helpful assistant."},
    {"role": "user", "content": "Write a short summary of MiMo-V2-Flash."},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(input_ids, max_new_tokens=128)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```


## MiMoV2FlashConfig

[[autodoc]] MiMoV2FlashConfig

## MiMoV2FlashModel

[[autodoc]] MiMoV2FlashModel
    - forward

## MiMoV2FlashForCausalLM

[[autodoc]] MiMoV2FlashForCausalLM
    - forward
