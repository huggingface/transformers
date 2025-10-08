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
*This model was released on 2025-04-05 and added to Hugging Face Transformers on 2025-04-05.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Llama4

[Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) introduces three new models: Scout, Maverick, and Behemoth. Llama 4 Scout is a 17B active parameter multimodal model with 16 experts, fitting on a single NVIDIA H100 GPU and featuring an exceptional 10M-token context window—surpassing Gemma 3, Gemini 2.0 Flash-Lite, and Mistral 3.1. Maverick, also 17B parameters but with 128 experts, outperforms GPT-4o and Gemini 2.0 Flash in multimodal benchmarks and rivals DeepSeek v3 in reasoning and coding while using less than half its parameters. Both models are distilled from Llama 4 Behemoth, a 288B active parameter model that currently exceeds GPT-4.5, Claude Sonnet 3.7, and Gemini 2.0 Pro on STEM benchmarks, marking it as Meta’s most powerful LLM to date.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

messages = [
    {"role": "user", "content": "How do plants create energy?"},
]

pipeline = pipeline(task="text-generation", model="meta-llama/Llama-4-Scout-17B-16E-Instruct",dtype="auto")
pipeline(messages, do_sample=False, max_new_tokens=200)
```

</hfoption>
<hfoption id="Llama4ForConditionalGeneration">

```py
from transformers import AutoTokenizer, Llama4ForConditionalGeneration
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")
model = Llama4ForConditionalGeneration.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", dtype="auto")


messages = [
    {"role": "user", "content": "How do plants create energy?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
print(tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:]))
```

</hfoption>
</hfoptions>

## Llama4Config

[[autodoc]] Llama4Config

## Llama4TextConfig

[[autodoc]] Llama4TextConfig

## Llama4VisionConfig

[[autodoc]] Llama4VisionConfig

## Llama4Processor

[[autodoc]] Llama4Processor

## Llama4ImageProcessorFast

[[autodoc]] Llama4ImageProcessorFast

## Llama4ForConditionalGeneration

[[autodoc]] Llama4ForConditionalGeneration
- forward

## Llama4ForCausalLM

[[autodoc]] Llama4ForCausalLM
- forward

## Llama4TextModel

[[autodoc]] Llama4TextModel
- forward

## Llama4ForCausalLM

[[autodoc]] Llama4ForCausalLM
- forward

## Llama4VisionModel

[[autodoc]] Llama4VisionModel
- forward
