<!--Copyright 2026 the HuggingFace Team. All rights reserved.

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
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-02-08.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# GlmMoeDsa

[GlmMoeDsa](https://huggingface.co/papers/2602.15763) (GLM-5) is a 744B-parameter mixture-of-experts model with 40B active parameters per token, using DeepSeek Sparse Attention (DSA) for efficient 200K-token context handling. It was trained entirely on Huawei Ascend chips and matches frontier-level performance on reasoning and long-context benchmarks.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="zai-org/GLM-5",
    dtype=torch.bfloat16,
)
pipe("The theory of relativity states that")
```

</hfoption>
<hfoption id="AutoModelForCausalLM">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-5")
model = AutoModelForCausalLM.from_pretrained(
    "zai-org/GLM-5",
    dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("The theory of relativity states that", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## GlmMoeDsaConfig

[[autodoc]] GlmMoeDsaConfig

## GlmMoeDsaModel

[[autodoc]] GlmMoeDsaModel
    - forward

## GlmMoeDsaForCausalLM

[[autodoc]] GlmMoeDsaForCausalLM
    - forward
