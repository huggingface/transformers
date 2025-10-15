<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-06-30 and added to Hugging Face Transformers on 2025-07-21 and contributed by [AntonV](https://huggingface.co/AntonV).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Ernie 4.5 Moe

# Ernie 4.5

[Ernie 4.5](https://ernie.baidu.com/blog/posts/ernie4.5/) introduces three major innovations. First, it uses Multimodal Heterogeneous MoE pre-training, jointly training on text and images through modality-isolated routing, router orthogonal loss, and multimodal token-balanced loss to ensure effective cross-modal learning. Second, it employs a scaling-efficient infrastructure with heterogeneous hybrid parallelism, FP8 mixed precision, recomputation strategies, and advanced quantization (4-bit/2-bit) to achieve high training and inference efficiency across hardware platforms. Finally, modality-specific post-training tailors models for language and vision tasks using Supervised Fine-Tuning, Direct Preference Optimization, and a new Unified Preference Optimization method.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="baidu/ERNIE-4.5-21B-A3B-PT", dtype="auto")
pipeline("Plants generate energy through a process known as  ")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("baidu/ERNIE-4.5-21B-A3B-PT")

messages = [{"role": "user", "content": "How do plants generate energy?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

outputs = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.3,)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Ernie4_5_MoeConfig

[[autodoc]] Ernie4_5_MoeConfig

## Ernie4_5_MoeModel

[[autodoc]] Ernie4_5_MoeModel
    - forward

## Ernie4_5_MoeForCausalLM

[[autodoc]] Ernie4_5_MoeForCausalLM
    - forward
    - generate
