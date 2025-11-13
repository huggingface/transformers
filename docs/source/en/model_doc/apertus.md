<!--Copyright 2025 The HuggingFace Team and the Swiss AI Initiative. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-09-02 and added to Hugging Face Transformers on 2025-08-28.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Apertus

[Apertus](https://www.swiss-ai.org) is a family of large language models from the Swiss AI Initiative.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="swiss-ai/Apertus-8B", dtype="auto")
pipeline("Plants generate energy through a process known as  ")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B")
model = ArceeForCausalLM.from_pretrained("swiss-ai/Apertus-8B", dtype="auto")

inputs = tokenizer("Plants generate energy through a process known as  ", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## ApertusConfig

[[autodoc]] ApertusConfig

## ApertusModel

[[autodoc]] ApertusModel
    - forward

## ApertusForCausalLM

[[autodoc]] ApertusForCausalLM
    - forward

## ApertusForTokenClassification

[[autodoc]] ApertusForTokenClassification
    - forward
