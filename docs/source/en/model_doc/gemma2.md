
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
*This model was released on 2024-07-31 and added to Hugging Face Transformers on 2024-06-27 and contributed by [ArthurZ](https://huggingface.co/ArthurZ) and [pcuenq](https://huggingface.co/pcuenq).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Gemma2

[Gemma 2](https://huggingface.co/papers/2408.00118) is a new series of lightweight open-source models ranging from 2 billion to 27 billion parameters. It incorporates technical improvements to the Transformer architecture, including interleaved local-global attention and group-query attention. The 2B and 9B models are trained using knowledge distillation rather than standard next-token prediction. These models achieve top performance for their size and compete with models two to three times larger, and all versions are publicly released.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-9b", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Gemma2Config

[[autodoc]] Gemma2Config

## Gemma2Model

[[autodoc]] Gemma2Model
    - forward

## Gemma2ForCausalLM

[[autodoc]] Gemma2ForCausalLM
    - forward

## Gemma2ForSequenceClassification

[[autodoc]] Gemma2ForSequenceClassification
    - forward

## Gemma2ForTokenClassification

[[autodoc]] Gemma2ForTokenClassification
    - forward
