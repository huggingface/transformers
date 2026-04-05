<!--Copyright 2025 Sarvam AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-04-03.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# SarvamMoe

SarvamMoe is a mixture-of-experts language model from [Sarvam AI](https://www.sarvam.ai/). It uses 128 routed experts with 6 activated per token, plus a shared expert, with sigmoid-based scoring and group-limited top-k routing. The model features GQA (grouped-query attention) with QK normalization.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="sarvamai/sarvam-30b-fp8",
    dtype=torch.bfloat16,
)
pipe("The key to effective reasoning is")
```

</hfoption>
<hfoption id="AutoModelForCausalLM">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-30b-fp8")
model = AutoModelForCausalLM.from_pretrained(
    "sarvamai/sarvam-30b-fp8",
    dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("The key to effective reasoning is", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## SarvamMoeConfig

[[autodoc]] SarvamMoeConfig

## SarvamMoeModel

[[autodoc]] SarvamMoeModel
    - forward

## SarvamMoeForCausalLM

[[autodoc]] SarvamMoeForCausalLM
    - forward

## SarvamMoeForSequenceClassification

[[autodoc]] SarvamMoeForSequenceClassification
    - forward

## SarvamMoeForTokenClassification

[[autodoc]] SarvamMoeForTokenClassification
    - forward

## SarvamMoeForQuestionAnswering

[[autodoc]] SarvamMoeForQuestionAnswering
    - forward
