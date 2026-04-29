<!--
 Copyright 2025 Bytedance-Seed Ltd and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-08-22.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# SeedOss

SeedOss is ByteDance Seed's 36B-parameter dense language model with native 512K context length. It features flexible thinking budget control and strong reasoning and agent capabilities, trained on 12T tokens.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="ByteDance-Seed/Seed-OSS-36B-Base",
    dtype=torch.bfloat16,
)
pipe("The most important factor in language model training is")
```

</hfoption>
<hfoption id="AutoModelForCausalLM">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ByteDance-Seed/Seed-OSS-36B-Base")
model = AutoModelForCausalLM.from_pretrained(
    "ByteDance-Seed/Seed-OSS-36B-Base",
    dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("The most important factor in language model training is", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## SeedOssConfig

[[autodoc]] SeedOssConfig

## SeedOssModel

[[autodoc]] SeedOssModel
    - forward

## SeedOssForCausalLM

[[autodoc]] SeedOssForCausalLM
    - forward

## SeedOssForSequenceClassification

[[autodoc]] SeedOssForSequenceClassification
    - forward

## SeedOssForTokenClassification

[[autodoc]] SeedOssForTokenClassification
    - forward

## SeedOssForQuestionAnswering

[[autodoc]] SeedOssForQuestionAnswering
    - forward
