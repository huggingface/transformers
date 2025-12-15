<!--Copyright 2024 Mistral AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-09-11.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Ministral

[Ministral](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) is a 8B parameter language model that extends the Mistral architecture with alternating attention pattern. Unlike Mistral, that uses either full attention or sliding window attention consistently, Ministral alternates between full attention and sliding window attention layers, in a pattern of 1 full attention layer followed by 3 sliding window attention layers. This allows for a 128K context length support.

This architecture turns out to coincide with Qwen2, with the main difference being the presence of biases in attention projections in Ministral.

You can find the Ministral checkpoints under the [Mistral AI](https://huggingface.co/mistralai) organization.

## Usage

The example below demonstrates how to use Ministral for text generation:

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Ministral-8B-Instruct-2410", torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410")

>>> messages = [
...     {"role": "user", "content": "What is your favourite condiment?"},
...     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
...     {"role": "user", "content": "Do you have mayonnaise recipes?"}
... ]

>>> model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

>>> generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"Mayonnaise can be made as follows: (...)"
```

## MinistralConfig

[[autodoc]] MinistralConfig

## MinistralModel

[[autodoc]] MinistralModel
    - forward

## MinistralForCausalLM

[[autodoc]] MinistralForCausalLM
    - forward

## MinistralForSequenceClassification

[[autodoc]] MinistralForSequenceClassification
    - forward

## MinistralForTokenClassification

[[autodoc]] MinistralForTokenClassification
    - forward

## MinistralForQuestionAnswering

[[autodoc]] MinistralForQuestionAnswering
    - forward
