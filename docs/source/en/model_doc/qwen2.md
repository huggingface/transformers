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
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Qwen2

[Qwen2](https://huggingface.co/papers/2407.10671) is a family of large language models (pretrained, instruction-tuned and mixture-of-experts) available in sizes from 0.5B to 72B parameters. The models are built on the Transformer architecture featuring enhancements like group query attention (GQA), rotary positional embeddings (RoPE), a mix of sliding window and full attention, and dual chunk attention with YARN for training stability. Qwen2 models support multiple languages and context lengths up to 131,072 tokens.

You can find all the official Qwen2 checkpoints under the [Qwen2](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f) collection.

> [!TIP]
> Click on the Qwen2 models in the right sidebar for more examples of how to apply Qwen2 to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line using the instruction-tuned models.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="Qwen/Qwen2-1.5B-Instruct",
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

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

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
# pip install -U flash-attn --no-build-isolation
transformers chat Qwen/Qwen2-7B-Instruct --dtype auto --attn_implementation flash_attention_2 --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 4-bits.

```python
# pip install -U flash-attn --no-build-isolation
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2"
)

inputs = tokenizer("The Qwen2 model family is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


## Notes

- Ensure your Transformers library version is up-to-date. Qwen2 requires Transformers>=4.37.0 for full support.

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
