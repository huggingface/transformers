<!--Copyright 2024 The Qwen Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-04-29 and added to Hugging Face Transformers on 2025-03-31.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Qwen3MoE

[Qwen3MoE](https://huggingface.co/papers/2505.09388) is the mixture-of-experts variant in the Qwen3 family, with 30.5B total parameters and 3.3B active parameters per token. It uses 128 routed experts with 8 activated per token across 48 layers, and supports up to 131K context with YaRN. See also the dense variant [Qwen3](qwen3).

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="Qwen/Qwen3-30B-A3B",
    dtype=torch.bfloat16,
)
pipe("The key to effective reasoning is")
```

</hfoption>
<hfoption id="AutoModelForCausalLM">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("The key to effective reasoning is", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Qwen3MoeConfig

[[autodoc]] Qwen3MoeConfig

## Qwen3MoeModel

[[autodoc]] Qwen3MoeModel
    - forward

## Qwen3MoeForCausalLM

[[autodoc]] Qwen3MoeForCausalLM
    - forward

## Qwen3MoeForSequenceClassification

[[autodoc]] Qwen3MoeForSequenceClassification
    - forward

## Qwen3MoeForTokenClassification

[[autodoc]] Qwen3MoeForTokenClassification
    - forward

## Qwen3MoeForQuestionAnswering

[[autodoc]] Qwen3MoeForQuestionAnswering
    - forward
