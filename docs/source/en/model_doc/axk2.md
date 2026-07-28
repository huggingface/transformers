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
*This model was contributed to Hugging Face Transformers on 2026-07-24.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# A.X-K2

[A.X-K2](https://huggingface.co/skt) is SK Telecom's flagship large language model. It is a
Mixture-of-Experts decoder built on the DeepSeek-V3.2 architecture — Multi-head Latent Attention (MLA)
with DeepSeek Sparse Attention (DSA) — plus three SK Telecom modifications:

- **Sparse Gated Attention (SGA)**: every layer runs a lightweight *lightning indexer* that scores each
  query against the keys and keeps only the top-`index_topk` positions, which become an additive sparse
  mask folded into the MLA attention. The indexer maintains its own key cache alongside the main KV
  cache (`DynamicIndexedLayer` / `StaticIndexedLayer`).
- **Gated RMSNorm**: `input_layernorm` (every layer) and `post_attention_layernorm` (MoE layers) are
  wrapped with a low-rank input-dependent sigmoid gate, `RMSNorm(x) * sigmoid(gate_mlp(RMSNorm(x)))`.
- **Attention output gate**: the attention output is multiplied by an input-dependent sigmoid gate
  (`g_proj`) before the output projection. In the released checkpoint this gate is fused into `q_b_proj`
  (vLLM layout) and split back out at load time by the weight converter.

Routing is plain (non-grouped) sigmoid top-k with a correction bias; the first layer is dense and the
rest are MoE (with a shared expert).

> [!TIP]
> A.X-K2 relies on an explicit additive sparse mask, so it runs under the `eager` and `sdpa` attention
> implementations (`attn_implementation="sdpa"` is the default and recommended backend).

The example below shows how to generate text with [`Pipeline`] or the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(task="text-generation", model="skt/A.X-K2")

print(pipe("대한민국의 수도는", max_new_tokens=32)[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("skt/A.X-K2")
model = AutoModelForCausalLM.from_pretrained("skt/A.X-K2", device_map="auto")

inputs = tokenizer("대한민국의 수도는", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## AXK2Config

[[autodoc]] AXK2Config

## AXK2Model

[[autodoc]] AXK2Model
    - forward

## AXK2ForCausalLM

[[autodoc]] AXK2ForCausalLM
    - forward

## AXK2ForSequenceClassification

[[autodoc]] AXK2ForSequenceClassification
    - forward

## AXK2ForTokenClassification

[[autodoc]] AXK2ForTokenClassification
    - forward
