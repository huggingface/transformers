<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was released on 2025-07-09 and added to Hugging Face Transformers on 2025-09-18.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# FlexOlmo

[FlexOlmo](https://huggingface.co/papers/2507.07024) is a new mixture-of-experts (MoE) language model architecture designed to allow distributed training on closed datasets without sharing raw data, and flexible inference where experts can be selectively included or excluded. Each expert is trained independently on its own dataset and later integrated using a domain-informed routing mechanism, avoiding joint training. The model is trained on FlexMix, a corpus combining public data with seven domain-specific closed sets, and is evaluated at scales up to 37B parameters (20B active) across 31 tasks. Results show a 41% relative performance gain from combining public and private experts, a 10.1% improvement over prior model-merging methods, and even better performance than standard MoE baselines trained under unrestricted data access, providing a practical solution for regulated or sensitive data contexts.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="allenai/FlexOlmo-7x7B-1T", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/FlexOlmo-7x7B-1T")

model = AutoModelForCausalLM.from_pretrained("allenai/FlexOlmo-7x7B-1T", dtype="auto",)
inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## FlexOlmoConfig

[[autodoc]] FlexOlmoConfig

## FlexOlmoForCausalLM

[[autodoc]] FlexOlmoForCausalLM

## FlexOlmoModel

[[autodoc]] FlexOlmoModel
    - forward

## FlexOlmoPreTrainedModel

[[autodoc]] FlexOlmoPreTrainedModel
    - forward
