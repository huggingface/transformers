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

<div style="float: right;">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
</div>

# Qwen2MoE


[Qwen2MoE](https://huggingface.co/papers/2407.10671) is a Mixture-of-Experts (MoE) variant of [Qwen2](./qwen2), available as a base model and an aligned chat model. It uses SwiGLU activation, group query attention and a mixture of sliding window attention and full attention. The tokenizer can also be adapted to multiple languages and codes.

The MoE architecture uses upcyled models from the dense language models. For example, Qwen1.5-MoE-A2.7B is upcycled from Qwen-1.8B. It has 14.3B parameters but only 2.7B parameters are activated during runtime.

You can find all the original checkpoints in the [Qwen1.5](https://huggingface.co/collections/Qwen/qwen15-65c0a2f577b1ecb76d786524) collection.

> [!TIP]
> Click on the Qwen2MoE models in the right sidebar for more examples of how to apply Qwen2MoE to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="Qwen/Qwen1.5-MoE-A2.7B",
    dtype=torch.bfloat16,
    device_map=0
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about the Qwen2 model family."},
]
outputs = pipe(messages, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"][-1]['content'])
```
</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat")

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

generated_ids = model.generate(
    model_inputs.input_ids,
    cache_implementation="static",
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```
</hfoption> 
<hfoption id="transformers CLI">
```bash
transformers chat Qwen/Qwen1.5-MoE-A2.7B-Chat --dtype auto --attn_implementation flash_attention_2
```
</hfoption>
 </hfoptions> 


Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 8-bits.

```python
# pip install -U flash-attn --no-build-isolation
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2"
)

inputs = tokenizer("The Qwen2 model family is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

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
