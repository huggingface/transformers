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
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-09-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# OLMo3

Olmo3 is an improvement on [OLMo2](./olmo2). More details will be released on *soon*.

> [!TIP]
> Click on the OLMo3 models in the right sidebar for more examples of how to apply OLMo3 to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`] and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="allenai/TBA",
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
    "allenai/TBA"
)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/TBA",
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
echo -e "Plants create energy through a process known as" | transformers run --task text-generation --model allenai/TBA --device 0
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
    "allenai/TBA"
)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/TBA",
    quantization_config=torchao_config,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_length=50, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))

```

## Notes

- Load specific intermediate checkpoints by adding the `revision` parameter to [`~PreTrainedModel.from_pretrained`].

    ```py
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("allenai/TBA", revision="stage1-step140000-tokens294B")
    ```

## Olmo3Config

[[autodoc]] Olmo3Config

## Olmo3ForCausalLM

[[autodoc]] Olmo3ForCausalLM

## Olmo3Model

[[autodoc]] Olmo3Model
    - forward

## Olmo3PreTrainedModel

[[autodoc]] Olmo3PreTrainedModel
    - forward
