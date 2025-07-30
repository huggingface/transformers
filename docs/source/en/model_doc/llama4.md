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

# Llama4


<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

Llama 4, developed by Meta, introduces a new auto-regressive Mixture-of-Experts (MoE) architecture.
This generation includes two models:
- The highly capable Llama 4 Maverick with 17B active parameters out of ~400B total, with 128 experts.
- The efficient Llama 4 Scout also  has 17B active parameters out of ~109B total, using just 16 experts.
-
Both models leverage early fusion for native multimodality, enabling them to process text and image inputs.
Maverick and Scout are both trained on up to 40 trillion tokens on data encompassing 200 languages
(with specific fine-tuning support for 12 languages including Arabic, Spanish, German, and Hindi).

For deployment, Llama 4 Scout is designed for accessibility, fitting on a single server-grade GPU via
on-the-fly 4-bit or 8-bitint4 quantization, while Maverick is available in BF16 and FP8 formats.
These models are released under the custom Llama 4 Community License Agreement, available on the model repositories.

You can find all the original Llama checkpoints under the [meta-llama](https://huggingface.co/meta-llama) organization.

> [!TIP]
> The Llama 4 family of models comes in two flavors: 109B, and 402B parameters. Both of these flavors are extremely
> large and won't fit on your run-of-the-mill device. See below for some examples to reduce the memory usage of the
> model.
>
> For the download to be faster and more resilient, we recommend installing the `hf_xet` dependency as followed:
> `pip install transformers[hf_xet]`

The examples below demonstrates how to generate with [`Pipeline`] or the [`AutoModel`]. We additionally add an example
showcasing how to toggle the right attributes to enable very long-context generations, as some flavors of Llama 4
have context lengths going up to 10 million tokens.


<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

messages = [
    {"role": "user", "content": "what is the recipe of mayonnaise?"},
]

pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    dtype=torch.bfloat16
)

output = pipe(messages, do_sample=False, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
```

</hfoption>
<hfoption id="AutoModel - Text only">

```py
from transformers import AutoTokenizer, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16
)

outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
print(outputs[0])
```

</hfoption>
<hfoption id="AutoModel - Multimodal">

```py
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
)

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": img_url},
            {"type": "text", "text": "Describe this image in two sentences."},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
)

response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print(response)
```

</hfoption>
<hfoption id="AutoModel - Multimodal with multiple images">

```py
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
)

url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url1},
            {"type": "image", "url": url2},
            {"type": "text", "text": "Can you describe how these two images are similar, and how they differ?"},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
)

response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print(response)
```

</hfoption>
<hfoption id="AutoModel - Long context">

Beware: the example below uses both `device_map="auto"` and flex-attention.
Please use `torchrun` to run this example in tensor-parallel mode.

We will work to enable running with `device_map="auto"` and flex-attention without
tensor-parallel in the future.

```py
from transformers import Llama4ForConditionalGeneration, AutoTokenizer
import torch
import time

file = "very_long_context_prompt.txt"
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

with open(file, "r") as f:
    very_long_text = "\n".join(f.readlines())

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flex_attention",
    dtype=torch.bfloat16
)

messages = [
    {"role": "user", "content": f"Look at the following texts: [{very_long_text}]\n\n\n\nWhat are the books, and who wrote them? Make me a nice list."},
]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

torch.cuda.synchronize()
start = time.time()
out = model.generate(
    input_ids.to(model.device),
    prefill_chunk_size=2048*8,
    max_new_tokens=300,
    cache_implementation="hybrid",
)
print(time.time()-start)
print(tokenizer.batch_decode(out[:, input_ids.shape[-1]:]))
print(f"{torch.cuda.max_memory_allocated(model.device) / 1024**3:.2f} GiB")
```

</hfoption>
</hfoptions>

## Efficiency; how to get the best out of llama 4

### The Attention methods

Updating the default attention function can significantly improve compute performance as well as memory usage. Refer to the [Attention Interface](../attention_interface) overview for an in-depth explanation of our interface.

As of release, the Llama 4 model supports the following attention methods: `eager`, `flex_attention`, `sdpa`. We recommend using `flex_attention` for best results.
Switching attention mechanism is done at the model initialization step:


<hfoptions id="Attention">
<hfoption id="Flex Attention">

Setting Flex Attention ensures the best results with the very long context the model can handle.

> [!TIP] Beware: the example below uses both `device_map="auto"` and flex-attention.
> Please use `torchrun` to run this example in tensor-parallel mode.
>
> We will work to enable running with `device_map="auto"` and flex-attention without
> tensor-parallel in the future.

```py
from transformers import Llama4ForConditionalGeneration
import torch

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="flex_attention",
    device_map="auto",
    dtype=torch.bfloat16,
)
```
</hfoption>
<hfoption id="SDPA">
The `sdpa` attention method is generally more compute-efficient than the `eager` method.

```py
from transformers import Llama4ForConditionalGeneration
import torch

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="sdpa",
    device_map="auto",
    dtype=torch.bfloat16,
)
```
</hfoption>
<hfoption id="Eager">
The `eager` attention method is set by default, so no need for anything different when loading the model:

```py
from transformers import Llama4ForConditionalGeneration
import torch

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
)
```
</hfoption>
</hfoptions>


### Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for available quantization backends.
At time of release, both FBGEMM and LLM-Compressor are supported; more quantization methods will be supported in the days that follow the release.

See below for examples using both:



Here is an example loading an BF16 model in FP8 using the FBGEMM approach:

<hfoptions id="Quantization">
<hfoption id="FBGEMM">

```python
from transformers import AutoTokenizer, Llama4ForConditionalGeneration, FbgemmFp8Config
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=FbgemmFp8Config()
)

outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
print(outputs[0])
```

</hfoption>
<hfoption id="LLM-Compressor">

To use the LLM-Compressor technique, we recommend leveraging the pre-quantized FP8 checkpoint available with the release:

```python
from transformers import AutoTokenizer, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    tp_plan="auto",
    dtype=torch.bfloat16,
)

outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
print(outputs[0])
```
</hfoption>
</hfoptions>

### Offloading

Enabling CPU-offloading means that components of the model might be moved to CPU instead of GPU in case the GPU-memory available isn't sufficient to load the entire model.
At inference, different components will be loaded/unloaded from/to the GPU on the fly. This ensures that the model can be loaded on smaller machines as long as the CPU-memory is sufficient.
However, this also slows down inference as it adds communication overhead.

In order to enable CPU-offloading, you simply need to specify the `device_map` to `auto` at model load:

```py
from transformers import Llama4ForConditionalGeneration
import torch

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
)
```

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
