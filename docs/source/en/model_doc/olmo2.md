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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# OLMo2
[OLMo2](https://huggingface.co/papers/2501.00656) is the next generation of fully open language models by AllenAI.
It is the successor to [OLMo](https://huggingface.co/docs/transformers/main/en/model_doc/olmo), and improves upon the original family of models
by changing the architecture and training recipes of the original models. This includes using 
RMSNorm rather than nonparametric LayerNorm, normalizing the outputs of the attention/feedforward layers rather
than the inputs, and using a new, specialized data mix called Dolmino Mix 1124 for the mid-training stage.

You can find all the original OLMo2 checkpoints under the [OLMo2](https://huggingface.co/collections/allenai/olmo-2-674117b93ab84e98afc72edc) collection.

> [!TIP]
> Click on the OLMo2 models in the right sidebar for more examples of how to apply OLMo2 to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`] and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="allenai/OLMo-2-0425-1B",
    torch_dtype=torch.float16,
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
    "allenai/OLMo-2-0425-1B"
)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-2-0425-1B",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers-cli">

```bash
echo -e "Plants create energy through a process known as" | transformers-cli run --task text-generation --model allenai/OLMo-2-0425-1B --device 0
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
    "allenai/OLMo-2-0425-1B"
)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-2-0425-1B",
    quantization_config=torchao_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))

```


## Notes

- OLMo2 requires Transformers v4.48 or higher.
- Specific model revisions can be loaded with HuggingFace by adding the `revision` argument. For pretraining, the naming convention is `stage1-stepXXX-tokensYYYB`, and for checkpoints with ingredients of the soup, the naming convention is `stage2-ingredientN-stepXXX-tokensYYYB`.

```py
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B", revision="stage1-step140000-tokens294B")
```


## Olmo2Config

[[autodoc]] Olmo2Config

## Olmo2Model

[[autodoc]] Olmo2Model
    - forward

## Olmo2ForCausalLM

[[autodoc]] Olmo2ForCausalLM
    - forward
