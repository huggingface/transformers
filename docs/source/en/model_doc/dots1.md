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
*This model was released on 2025-06-06 and added to Hugging Face Transformers on 2025-06-25.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# dots.llm1

[dots.llm1](https://huggingface.co/papers/2506.05767) is a 142B-parameter mixture-of-experts model that activates 14B parameters per token, using top-6-of-128 routed experts plus 2 shared experts. It delivers performance on par with Qwen2.5-72B while significantly reducing training and inference costs. Notably, no synthetic data was used during pretraining.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="rednote-hilab/dots.llm1.base",
    dtype=torch.bfloat16,
)
pipe("The advantage of mixture-of-experts models is")
```

</hfoption>
<hfoption id="AutoModelForCausalLM">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("rednote-hilab/dots.llm1.base")
model = AutoModelForCausalLM.from_pretrained(
    "rednote-hilab/dots.llm1.base",
    dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("The advantage of mixture-of-experts models is", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Dots1Config

[[autodoc]] Dots1Config

## Dots1Model

[[autodoc]] Dots1Model
    - forward

## Dots1ForCausalLM

[[autodoc]] Dots1ForCausalLM
    - forward
