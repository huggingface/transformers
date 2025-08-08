<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>


# Cohere2

[Cohere Command R7B](https://cohere.com/blog/command-r7b) is an open weights research release of a 7B billion parameter model. It is a multilingual model trained on 23 languages and has a context window of 128k. The model features three layers with sliding window attention and ROPE for efficient local context modeling and relative positional encoding. A fourth layer uses global attention without positional embeddings, enabling unrestricted token interactions across the entire sequence.

This model is optimized for speed, cost-performance, and compute resources.

You can find all the original Command-R checkpoints under the [Command Models](https://huggingface.co/collections/CohereForAI/command-models-67652b401665205e17b192ad) collection.


> [!TIP]
> Click on the Cohere models in the right sidebar for more examples of how to apply Cohere to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModel`] class, and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation", 
    model="CohereLabs/c4ai-command-r7b-12-2024",
    dtype=torch.float16,
    device_map=0
)

messages = [
    {"role": "user", "content": "Hello, can you please help me book a hotel in Japan?"},
]
pipeline(messages)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("CohereLabs/c4ai-command-r7b-12-2024")
model = AutoModelForCausalLM.from_pretrained(
    "CohereLabs/c4ai-command-r7b-12-2024", 
    dtype=torch.float16, 
    device_map="auto", 
    attn_implementation="sdpa"
)

# format message with the Command-R chat template
messages = [{"role": "user", "content": "Hello, can you please help me book a hotel in Japan?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
output = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    cache_implementation="static",
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
# pip install -U flash-attn --no-build-isolation
transformers-cli chat CohereLabs/c4ai-command-r7b-12-2024 --dtype auto --attn_implementation flash_attention_2
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview.md) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes.md) to quantize the weights to 4-bits.

```python
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("CohereLabs/c4ai-command-r7b-12-2024")
model = AutoModelForCausalLM.from_pretrained(
    "CohereLabs/c4ai-command-r7b-12-2024", 
    dtype=torch.float16, 
    device_map="auto", 
    quantization_config=bnb_config, 
    attn_implementation="sdpa"
)

# format message with the Command-R chat template
messages = [{"role": "user", "content": "Hello, can you please help me book a hotel in Japan?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
output = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    cache_implementation="static",
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Cohere2Config

[[autodoc]] Cohere2Config

## Cohere2Model

[[autodoc]] Cohere2Model
    - forward


## Cohere2ForCausalLM

[[autodoc]] Cohere2ForCausalLM
    - forward


