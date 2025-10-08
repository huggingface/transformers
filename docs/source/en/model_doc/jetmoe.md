<!--Copyright 2024 JetMoe team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-06-07 and added to Hugging Face Transformers on 2024-05-14 and contributed by [YikangS](https://huggingface.co/YikangS).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# JetMoe

[JetMoE](https://huggingface.co/papers/2404.07413) is a large language model trained on 1.25 trillion tokens using 30,000 H100 GPU hours at a cost under $0.1M. It employs a sparsely-gated mixture-of-experts (SMoE) architecture, where both attention and feedforward layers are sparsely activated, resulting in 8B parameters but only 2B active per token—cutting inference compute by about 70% compared to Llama2-7B. Despite its low-cost training, JetMoE-8B outperforms Llama2-7B, and its chat variant surpasses Llama2-13B-Chat. The model was trained entirely on open-source data and code, with full training details and weights publicly released to support open research and collaboration.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="jetmoe/jetmoe-8b", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b")
model = AutoModelForCausalLM.from_pretrained("jetmoe/jetmoe-8b", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## JetMoeConfig

[[autodoc]] JetMoeConfig

## JetMoeModel

[[autodoc]] JetMoeModel
    - forward

## JetMoeForCausalLM

[[autodoc]] JetMoeForCausalLM
    - forward

## JetMoeForSequenceClassification

[[autodoc]] JetMoeForSequenceClassification
    - forward
