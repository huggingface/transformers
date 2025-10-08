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
*This model was released on 2024-08-23 and added to Hugging Face Transformers on 2024-09-20 and contributed by [mayank-mishra](https://huggingface.co/mayank-mishra).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# GraniteMoe

[GraniteMoe](https://huggingface.co/papers/2408.13359) is a 3B sparse Mixture-of-Experts (sMoE) language model trained using the Power learning rate scheduler. It activates 800M parameters per token and demonstrates competitive performance compared to dense models with twice the active parameters across various benchmarks such as natural language multi-choice, code generation, and math reasoning. The Power scheduler, which is agnostic to batch size and number of training tokens, achieves consistent performance across different model sizes and architectures when combined with Maximum Update Parameterization (mup).

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="ibm/PowerMoE-3b", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ibm/PowerMoE-3b")
model = AutoModelForCausalLM.from_pretrained("ibm/PowerMoE-3b", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## GraniteMoeConfig

[[autodoc]] GraniteMoeConfig

## GraniteMoeModel

[[autodoc]] GraniteMoeModel
    - forward

## GraniteMoeForCausalLM

[[autodoc]] GraniteMoeForCausalLM
    - forward

