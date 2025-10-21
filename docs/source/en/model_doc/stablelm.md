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
*This model was released on 2023-09-05 and added to Hugging Face Transformers on 2024-02-14.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# StableLM

[StableLM](https://huggingface.co/papers/2402.17834) is a new-generation language model with both base and instruction-tuned versions, whose weights are publicly available on Hugging Face. The model was trained with a detailed data and training procedure, and it achieves strong performance on zero- and few-shot tasks, multilingual benchmarks, and multi-turn dialogue MT evaluations. At under 2 billion parameters, it was the leading open model in its size class when released. The report also includes throughput benchmarks on edge devices and performance metrics for several open-source quantized checkpoints.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="stabilityai/stablelm-3b-4e1t", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-3b-4e1t", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- The architecture is similar to LLaMA but with key differences: RoPE applied to 25% of head embedding dimensions, LayerNorm instead of RMSNorm, and optional QKV bias terms.
- StableLM 3B 4E1T-based models use the same tokenizer as [`GPTNeoXTokenizerFast`].

## StableLmConfig

[[autodoc]] StableLmConfig

## StableLmModel

[[autodoc]] StableLmModel
    - forward

## StableLmForCausalLM

[[autodoc]] StableLmForCausalLM
    - forward

## StableLmForSequenceClassification

[[autodoc]] StableLmForSequenceClassification
    - forward

## StableLmForTokenClassification

[[autodoc]] StableLmForTokenClassification
    - forward
