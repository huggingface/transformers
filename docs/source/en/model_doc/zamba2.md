<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-11-22 and added to Hugging Face Transformers on 2025-01-27 and contributed by [pglo](https://huggingface.co/pglo).*
# Zamba2

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

[Zamba2](https://huggingface.co/papers/2411.15242) introduces hybrid Mamba2-transformer models with 1.2B, 2.7B, and 7.4B parameters, delivering state-of-the-art performance among open-weight models of similar scale. Compared to previous work (Zamba1-7B), the new models feature architectural improvements, optimized training strategies, and training on up to three trillion tokens. They achieve significant efficiency gains in inference latency, throughput, and memory usage. All models, along with instruction-tuned variants and the Zyda-2 pretraining dataset, are released as open source on Hugging Face.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Zyphra/Zamba2-1.2B", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-1.2B")
model = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba2-1.2B", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Zamba2Config

[[autodoc]] Zamba2Config

## Zamba2Model

[[autodoc]] Zamba2Model
    - forward

## Zamba2ForCausalLM

[[autodoc]] Zamba2ForCausalLM
    - forward

## Zamba2ForSequenceClassification

[[autodoc]] transformers.Zamba2ForSequenceClassification
    - forward
