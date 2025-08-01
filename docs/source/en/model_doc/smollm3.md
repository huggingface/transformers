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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# SmolLM3

SmolLM3 is a fully open, compact language model designed for efficient deployment while maintaining strong performance. It uses a Transformer decoder architecture with Grouped Query Attention (GQA) to reduce the kv cache, and no RoPE, enabling improved performance on long-context tasks. It is trained using a multi-stage training approach on high-quality public datasets across web, code, and math domains. The model is multilingual and supports very large context lengths. The instruct variant is optimized for reasoning and tool use.

> [!TIP]
> Click on the SmolLM3 models in the right sidebar for more examples of how to apply SmolLM3 to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line using the instruction-tuned models.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="HuggingFaceTB/SmolLM3-3B",
    dtype=torch.bfloat16,
    device_map=0
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about yourself."},
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
    "HuggingFaceTB/SmolLM3-3B",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

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
transformers chat HuggingFaceTB/SmolLM3-3B --dtype auto --attn_implementation flash_attention_2 --device 0
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

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM3-3B",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2"
)

inputs = tokenizer("Gravity is the force", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


## Notes

- Ensure your Transformers library version is up-to-date. SmolLM3 requires Transformers>=4.53.0 for full support.

## SmolLM3Config

[[autodoc]] SmolLM3Config

## SmolLM3Model

[[autodoc]] SmolLM3Model
    - forward

## SmolLM3ForCausalLM

[[autodoc]] SmolLM3ForCausalLM
    - forward

## SmolLM3ForSequenceClassification

[[autodoc]] SmolLM3ForSequenceClassification
    - forward

## SmolLM3ForTokenClassification

[[autodoc]] SmolLM3ForTokenClassification
    - forward

## SmolLM3ForQuestionAnswering

[[autodoc]] SmolLM3ForQuestionAnswering
    - forward
