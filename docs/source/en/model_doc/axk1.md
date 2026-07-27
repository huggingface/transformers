<!--Copyright 2026 SK Telecom and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-07-23.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# A.X-K1

[A.X-K1](https://huggingface.co/skt) is SK Telecom's Mixture-of-Experts large language model. It is
built on the DeepSeek-V3 architecture — Multi-head Latent Attention (MLA) with a grouped sigmoid
top-k MoE and a shared expert — with one SK Telecom modification: an extra **`post_mlp_layernorm`**
applied to the MoE block output before the residual add. The first layer is dense and the rest are
MoE.

Because attention is standard (dense) MLA, A.X-K1 runs under all attention backends (FlashAttention-2,
SDPA, and eager).

The example below shows how to generate text with [`Pipeline`] or the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="skt/A.X-K1",
)

print(pipe("대한민국의 수도는", max_new_tokens=32)[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("skt/A.X-K1")
model = AutoModelForCausalLM.from_pretrained(
    "skt/A.X-K1",
    device_map="auto",
)

inputs = tokenizer("대한민국의 수도는", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## AXK1Config

[[autodoc]] AXK1Config

## AXK1Model

[[autodoc]] AXK1Model
    - forward

## AXK1ForCausalLM

[[autodoc]] AXK1ForCausalLM
    - forward

## AXK1ForSequenceClassification

[[autodoc]] AXK1ForSequenceClassification
    - forward

## AXK1ForTokenClassification

[[autodoc]] AXK1ForTokenClassification
    - forward
