<!--Copyright 2026 the HuggingFace Team. All rights reserved.
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
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-01-19.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# OLMo 3.5 Hybrid

OLMo 3.5 Hybrid is a hybrid architecture model from Ai2 that combines standard transformer attention layers with linear attention layers using the Gated Deltanet. This hybrid approach aims to improve efficiency while maintaining model quality by interleaving full attention layers with linear attention layers.

> [!TIP]
> For optimal performance, install the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) library. The model will work without it using a PyTorch fallback, but FLA provides significant speedups for the linear attention layers.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`] and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">
```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="allenai/OLMo-3.5-1B-Hybrid",
    torch_dtype=torch.bfloat16,
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
    "allenai/OLMo-3.5-1B-Hybrid"
)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-3.5-1B-Hybrid",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">
```bash
echo -e "Plants create energy through a process known as" | transformers-cli run --task text-generation --model allenai/OLMo-3.5-1B-Hybrid --device 0
```


## Notes

- For best performance with linear attention layers, install [flash-linear-attention](https://github.com/fla-org/flash-linear-attention):
```bash
  pip install flash-linear-attention
```

- The model uses a custom cache (`Olmo3_5HybridDynamicCache`) that handles both KV cache for attention layers and recurrent state for linear attention layers.

## Olmo3_5HybridConfig

[[autodoc]] Olmo3_5HybridConfig

## Olmo3_5HybridModel

[[autodoc]] Olmo3_5HybridModel
    - forward

## Olmo3_5HybridForCausalLM

[[autodoc]] Olmo3_5HybridForCausalLM
    - forward