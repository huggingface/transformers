<!--Copyright 2025 The ZhipuAI Inc. and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-01-19 and added to Hugging Face Transformers on 2026-01-13.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Glm4MoeLite

Glm4MoeLite (GLM-4.7-Flash) is a 30B-parameter mixture-of-experts model with approximately 3B active parameters per token, designed for lightweight deployment that balances performance and efficiency. It is part of the GLM-4.7 family and supports interleaved thinking capabilities.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="zai-org/GLM-4.7-Flash",
    dtype=torch.bfloat16,
)
pipe("The key to efficient language models is")
```

</hfoption>
<hfoption id="AutoModelForCausalLM">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")
model = AutoModelForCausalLM.from_pretrained(
    "zai-org/GLM-4.7-Flash",
    dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("The key to efficient language models is", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Glm4MoeLiteConfig

[[autodoc]] Glm4MoeLiteConfig

## Glm4MoeLiteModel

[[autodoc]] Glm4MoeLiteModel
    - forward

## Glm4MoeLiteForCausalLM

[[autodoc]] Glm4MoeLiteForCausalLM
    - forward
