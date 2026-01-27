<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was released on 2025-07-09 and added to Hugging Face Transformers on 2025-09-18.*
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# FlexOlmo

[FlexOlmo](https://huggingface.co/papers/2507.07024) is a new class of language models (LMs) that supports (1) distributed training without data sharing, where different model parameters are independently trained on closed datasets, and (2) data-flexible inference, where these parameters along with their associated data can be flexibly included or excluded from model inferences with no further training. FlexOlmo employs a mixture-of-experts (MoE) architecture where each expert is trained independently on closed datasets and later integrated through a new domain-informed routing without any joint training. FlexOlmo is trained on FlexMix, a corpus we curate comprising publicly available datasets alongside seven domain-specific sets, representing realistic approximations of closed sets.

You can find all the original FlexOlmo checkpoints under the [FlexOlmo](https://huggingface.co/collections/allenai/flexolmo-68471177a386b6e20a54c55f) collection.

> [!TIP]
> Click on the FlexOlmo models in the right sidebar for more examples of how to apply FlexOlmo to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`] and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="allenai/FlexOlmo-7x7B-1T",
    dtype=torch.bfloat16,
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
    "allenai/FlexOlmo-7x7B-1T"
)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/FlexOlmo-7x7B-1T",
    dtype=torch.bfloat16,
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
echo -e "Plants create energy through a process known as" | transformers run --task text-generation --model allenai/FlexOlmo-7x7B-1T --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to 4-bits.

```py

#pip install torchao
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

torchao_config = TorchAoConfig(
    "int4_weight_only",
    group_size=128
)

tokenizer = AutoTokenizer.from_pretrained(
    "allenai/FlexOlmo-7x7B-1T"
)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/FlexOlmo-7x7B-1T",
    quantization_config=torchao_config,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_length=50, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))

```

## FlexOlmoConfig

[[autodoc]] FlexOlmoConfig

## FlexOlmoForCausalLM

[[autodoc]] FlexOlmoForCausalLM

## FlexOlmoModel

[[autodoc]] FlexOlmoModel
    - forward

## FlexOlmoPreTrainedModel

[[autodoc]] FlexOlmoPreTrainedModel
    - forward
