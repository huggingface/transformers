<!--Copyright 2026 The HuggingFace Team. All rights reserved.
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-12-15 and added to Hugging Face Transformers on 2026-03-02.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# NemotronH

[NemotronH](https://huggingface.co/papers/2504.03624) is a hybrid architecture combining attention and state-space layers for efficient long-context language modeling. It interleaves Mamba2 and transformer blocks, using a fixed ratio to balance expressiveness with linear-time sequence processing.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="nvidia/Nemotron-H-8B-Reasoning-128K",
    dtype=torch.bfloat16,
)
pipe("Plants create energy through a process known as")
```

</hfoption>
<hfoption id="AutoModelForCausalLM">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-H-8B-Reasoning-128K")
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-H-8B-Reasoning-128K",
    dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## NemotronHConfig

[[autodoc]] NemotronHConfig

## NemotronHModel

[[autodoc]] NemotronHModel
    - forward

## NemotronHForCausalLM

[[autodoc]] NemotronHForCausalLM
    - forward
