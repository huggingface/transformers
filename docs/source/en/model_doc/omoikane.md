<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-25.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Omoikane

Omoikane is a small hybrid mixture-of-experts language model. Omoikane-50M-A27M has 48.3M parameters, of which 27.1M
are active per token, and a 16k-token vocabulary. It is trained from scratch with MLX.

The decoder mixes two token mixers and two feed-forward blocks:

- Gated DeltaNet layers, with a short causal convolution on the queries, keys and values, L2-normalized queries and
  keys, and a gated RMSNorm on the output. Unlike Qwen3-Next, the recurrent state is read out with a scale of 1.
- Multi-head latent attention (MLA) at a few layers, as in DeepSeek-V3, but with rotary embeddings on rotated halves
  (`rotate_half`, not interleaved pairs) and a sigmoid gate on the attention output.
- Dense SwiGLU MLPs in the first and last layers, and mixture-of-experts blocks elsewhere: 16 routed experts with
  sigmoid scores, a correction bias, group-limited top-4 routing and one shared expert.

The checkpoints also contain a multi-token-prediction (MTP) module, which is used only in training. Loading ignores it.

The example below demonstrates how to generate text with [`Pipeline`] and [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(task="text-generation", model="Goekdeniz-Guelmez/Omoikane-50M-A27M-Base")
print(pipe("Plants create energy through a process known as", max_new_tokens=40))
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Goekdeniz-Guelmez/Omoikane-50M-A27M-Base")
model = AutoModelForCausalLM.from_pretrained("Goekdeniz-Guelmez/Omoikane-50M-A27M-Base")

inputs = tokenizer("Plants create energy through a process known as", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Notes

- The training checkpoints store the token-mixer and expert settings as nested dictionaries (`gdn_config`,
  `mla_config`, `moe_config`). [`OmoikaneConfig`] reads that format and converts it to its own fields.
- The checkpoints store one depthwise convolution per projection (`q_conv1d`, `k_conv1d`, `v_conv1d`). The model
  concatenates them into one `conv1d` on load, and splits them again on save.
- Install [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) and
  [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) for fast Gated DeltaNet layers on CUDA. Without them, the
  model uses a PyTorch implementation.
- Use left padding for batched generation.

## OmoikaneConfig

[[autodoc]] OmoikaneConfig

## OmoikaneModel

[[autodoc]] OmoikaneModel
    - forward

## OmoikaneForCausalLM

[[autodoc]] OmoikaneForCausalLM
    - forward
