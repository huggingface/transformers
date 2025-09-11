<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-02-01 and added to Hugging Face Transformers on 2024-04-17.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# OLMo
[OLMo](https://huggingface.co/papers/2402.00838) is a 7B-parameter dense language model. It uses SwiGLU activations, non-parametric layer normalization, rotary positional embeddings, and a BPE tokenizer that masks personally identifiable information. It is pretrained on [Dolma](https://huggingface.co/datasets/allenai/dolma), a 3T-token dataset. OLMo was released to provide complete transparency of not just the model weights but the training data, training code, and evaluation code to enable more research on language models.

You can find all the original OLMo checkpoints under the [OLMo](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778) collection.

> [!TIP]
> This model was contributed by [shanearora](https://huggingface.co/shanearora).
>
> Click on the OLMo models in the right sidebar for more examples of how to apply OLMo to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="allenai/OLMo-7B-hf",
    dtype=torch.float16,
    device=0,
)

result = pipe("Plants create energy through a process known as")
print(result)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "allenai/OLMo-7B-hf"
)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-7B-hf",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_length=50, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create energy through a process known as" | transformers run --task text-generation --model allenai/OLMo-7B-hf --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bits.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-7B-hf",
    attn_implementation="sdpa",
    dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf")

inputs = tokenizer("Bitcoin is", return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

output = model.generate(**inputs, max_length=64)

print(tokenizer.decode(output[0]))
```

## OlmoConfig

[[autodoc]] OlmoConfig

## OlmoModel

[[autodoc]] OlmoModel
    - forward

## OlmoForCausalLM

[[autodoc]] OlmoForCausalLM
    - forward
