<!--Copyright 2024 The Qwen Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-07-15 and added to Hugging Face Transformers on 2024-01-17.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Qwen2

[Qwen2](https://huggingface.co/papers/2407.10671) is a series introduces a range of large language and multimodal models from 0.5 to 72 billion parameters, including dense and Mixture-of-Experts architectures, improving on its predecessor Qwen1.5 and performing competitively with proprietary models. The flagship Qwen2-72B achieves strong benchmark results across language understanding, generation, coding, mathematics, and reasoning, while its instruction-tuned variant excels on instruction-following tasks. Qwen2 supports around 30 languages, demonstrating broad multilingual capability. Model weights and supplementary resources for quantization, fine-tuning, and deployment are openly available on Hugging Face, ModelScope, and GitHub.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2-1.5B", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- Update Transformers to version 4.37.0 or higher. Qwen2 requires `Transformers>=4.37.0` for full support.

## Qwen2Config

[[autodoc]] Qwen2Config

## Qwen2Tokenizer

[[autodoc]] Qwen2Tokenizer
    - save_vocabulary

## Qwen2TokenizerFast

[[autodoc]] Qwen2TokenizerFast

## Qwen2Model

[[autodoc]] Qwen2Model
    - forward

## Qwen2ForCausalLM

[[autodoc]] Qwen2ForCausalLM
    - forward

## Qwen2ForSequenceClassification

[[autodoc]] Qwen2ForSequenceClassification
    - forward

## Qwen2ForTokenClassification

[[autodoc]] Qwen2ForTokenClassification
    - forward

## Qwen2ForQuestionAnswering

[[autodoc]] Qwen2ForQuestionAnswering
    - forward

## Qwen2RMSNorm

[[autodoc]] Qwen2RMSNorm
