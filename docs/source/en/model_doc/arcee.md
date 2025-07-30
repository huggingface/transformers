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

# Arcee

Arcee is a decoder-only transformer model based on the Llama architecture with a key modification: it uses ReLU² (ReLU-squared) activation in the MLP blocks instead of SiLU, following recent research showing improved training efficiency with squared activations. This architecture is designed for efficient training and inference while maintaining the proven stability of the Llama design.

The Arcee model is architecturally similar to Llama but uses `x * relu(x)` in MLP layers for improved gradient flow and is optimized for efficiency in both training and inference scenarios.

> [!TIP]
> The Arcee model supports extended context with RoPE scaling and all standard transformers features including Flash Attention 2, SDPA, gradient checkpointing, and quantization support.

The example below demonstrates how to generate text with Arcee using [`Pipeline`] or the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="arcee-ai/AFM-4.5B",
    dtype=torch.float16,
    device=0
)

output = pipeline("The key innovation in Arcee is")
print(output[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, ArceeForCausalLM

tokenizer = AutoTokenizer.from_pretrained("arcee-ai/AFM-4.5B")
model = ArceeForCausalLM.from_pretrained(
    "arcee-ai/AFM-4.5B",
    dtype=torch.float16,
    device_map="auto"
)

inputs = tokenizer("The key innovation in Arcee is", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## ArceeConfig

[[autodoc]] ArceeConfig

## ArceeModel

[[autodoc]] ArceeModel
    - forward

## ArceeForCausalLM

[[autodoc]] ArceeForCausalLM
    - forward

## ArceeForSequenceClassification

[[autodoc]] ArceeForSequenceClassification
    - forward

## ArceeForQuestionAnswering

[[autodoc]] ArceeForQuestionAnswering
    - forward

## ArceeForTokenClassification

[[autodoc]] ArceeForTokenClassification
    - forward