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

# Qwen3

[Qwen3](https://huggingface.co/papers/2505.09388) is the dense model architecture in the Qwen3 family, available in sizes from 0.6B to 32B parameters. It supports both thinking mode (multi-step reasoning) and non-thinking mode, with seamless switching between the two. Qwen3 was trained on approximately 36T tokens covering 119 languages. See also the MoE variant [Qwen3MoE](qwen3_moe).

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
)
pipe("The key to effective reasoning is")
```

</hfoption>
<hfoption id="AutoModelForCausalLM">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("The key to effective reasoning is", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Qwen3Config

[[autodoc]] Qwen3Config

## Qwen3Model

[[autodoc]] Qwen3Model
    - forward

## Qwen3ForCausalLM

[[autodoc]] Qwen3ForCausalLM
    - forward

## Qwen3ForSequenceClassification

[[autodoc]] Qwen3ForSequenceClassification
    - forward

## Qwen3ForTokenClassification

[[autodoc]] Qwen3ForTokenClassification
    - forward

## Qwen3ForQuestionAnswering

[[autodoc]] Qwen3ForQuestionAnswering
    - forward
