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
*This model was released on 2024-07-15 and added to Hugging Face Transformers on 2024-03-27.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Qwen2MoE

[Qwen2MoE](https://huggingface.co/papers/2407.10671) is a series introduces a range of large language and multimodal models with 0.5 to 72 billion parameters, including dense and Mixture-of-Experts architectures. It outperforms most previous open-weight models and shows competitive results versus proprietary models across benchmarks in language understanding, generation, coding, mathematics, reasoning, and multilingual tasks. The flagship Qwen2-72B achieves high scores on MMLU, GPQA, HumanEval, GSM8K, and BBH, while its instruction-tuned variant excels on MT-Bench, Arena-Hard, and LiveCodeBench. Qwen2 supports around 30 languages and is openly available on Hugging Face and ModelScope, with resources for quantization, fine-tuning, deployment, and example code on GitHub.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen1.5-MoE-A2.7B", dtype="auto",)
messages = [ 
    {"role": "system", "content": "You are a plant biologist."}, 
    {"role": "user", "content": "Can you explain how plants create energy?"}, 
] 
pipeline(messages)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", dtype="auto")

messages = [ 
    {"role": "system", "content": "You are a plant biologist."}, 
    {"role": "user", "content": "Can you explain how plants create energy?"},
] 

inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Qwen2MoeConfig

[[autodoc]] Qwen2MoeConfig

## Qwen2MoeModel

[[autodoc]] Qwen2MoeModel
    - forward

## Qwen2MoeForCausalLM

[[autodoc]] Qwen2MoeForCausalLM
    - forward

## Qwen2MoeForSequenceClassification

[[autodoc]] Qwen2MoeForSequenceClassification
    - forward

## Qwen2MoeForTokenClassification

[[autodoc]] Qwen2MoeForTokenClassification
    - forward

## Qwen2MoeForQuestionAnswering

[[autodoc]] Qwen2MoeForQuestionAnswering
    - forward
