<!--Copyright 2026 The HuggingFace Team and the KORMo Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2025-10-10 and contributed to Hugging Face Transformers on 2026-06-05.*

# KORMo

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

## Overview

KORMo (Korean Open Reasoning Model) is a fully open bilingual (Korean-English) large language model introduced in
[KORMo: Korean Open Reasoning Model for Everyone](https://huggingface.co/papers/2510.09426). The model, training code
and training data are released openly.

Architecturally, KORMo follows the Llama decoder design (RMSNorm, grouped-query attention, rotary position embeddings
and a SwiGLU MLP). The only difference is naming: the two per-layer RMSNorms are called `pre_attention_layernorm` and
`pre_mlp_layernorm` (Llama calls them `input_layernorm` and `post_attention_layernorm`).

You can find the KORMo checkpoints under the [KORMo-Team](https://huggingface.co/KORMo-Team) organization.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="KORMo-Team/KORMo-10B-sft",
    dtype=torch.bfloat16,
    device_map="auto",
)
messages = [{"role": "user", "content": "대한민국의 수도는 어디인가요?"}]
print(pipe(messages, max_new_tokens=64)[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("KORMo-Team/KORMo-10B-sft")
model = AutoModelForCausalLM.from_pretrained(
    "KORMo-Team/KORMo-10B-sft", dtype=torch.bfloat16, device_map="auto"
)

messages = [{"role": "user", "content": "대한민국의 수도는 어디인가요?"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## KORMoConfig

[[autodoc]] KORMoConfig

## KORMoModel

[[autodoc]] KORMoModel
    - forward

## KORMoForCausalLM

[[autodoc]] KORMoForCausalLM
    - forward

## KORMoForSequenceClassification

[[autodoc]] KORMoForSequenceClassification
    - forward

## KORMoForTokenClassification

[[autodoc]] KORMoForTokenClassification
    - forward

## KORMoForQuestionAnswering

[[autodoc]] KORMoForQuestionAnswering
    - forward
