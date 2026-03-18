<!--Copyright (C) 2024 THL A29 Limited, a Tencent company and The HuggingFace Inc. team. All rights reserved..

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
    </div>
</div>

# HunYuanMoEV1

HunYuanMoEV1 is Tencent's mixture-of-experts language model with 80B total parameters and 13B active parameters per token. It uses fine-grained expert routing with Grouped Query Attention, supports 256K context length, and offers dual-mode reasoning (fast and slow thinking).

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="tencent/Hunyuan-A13B-Instruct",
    dtype=torch.bfloat16,
)
pipe("The future of artificial intelligence is")
```

</hfoption>
<hfoption id="AutoModelForCausalLM">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tencent/Hunyuan-A13B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "tencent/Hunyuan-A13B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("The future of artificial intelligence is", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## HunYuanMoEV1Config

[[autodoc]] HunYuanMoEV1Config

## HunYuanMoEV1Model

[[autodoc]] HunYuanMoEV1Model
    - forward

## HunYuanMoEV1ForCausalLM

[[autodoc]] HunYuanMoEV1ForCausalLM
    - forward

## HunYuanMoEV1ForSequenceClassification

[[autodoc]] HunYuanMoEV1ForSequenceClassification
    - forward
