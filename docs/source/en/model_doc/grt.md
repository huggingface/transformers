<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-01.*

<div class="flex flex-wrap space-x-1">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# GRT

GRT (Geometric Rotational Transformer) is a decoder-only language model that replaces the dense weight matrices of a
standard transformer with chains of [Givens rotations](https://en.wikipedia.org/wiki/Givens_rotation). There are no
`nn.Linear` layers anywhere in the model.

Every projection — attention Q/K/V/O, the feed-forward expansion and contraction, and the cross-head mixer — is a
sequence of stages that pair up adjacent dimensions, rotate each pair by a learned angle, and apply a per-dimension
scale. A butterfly permutation before each stage changes which dimensions are paired, so a chain of `K` stages mixes
information across the whole vector the way a dense matrix would, but with `O(D)` parameters per stage rather than
`O(D²)`.

Three further pieces follow from that design:

- **Phase-angle embeddings.** A token is stored as `hidden_size / 2` angles rather than `hidden_size` free values,
  which places every token on the same hypersphere. A per-token log magnitude lets the model move tokens off it.
- **Content-dependent rotations.** Each stage derives an angle offset from the content of the vector it is rotating,
  using Clifford-algebra features (pair energy, both vector components, and a cross-pair bivector). This is what
  supplies the input dependence a dense matrix would otherwise provide.
- **Phase-preserving norm and activation.** `PearlNorm` normalizes pair radii and `RadialGELU` gates pair magnitudes;
  both leave every phase angle untouched.

The language modeling head is a cosine similarity between the final hidden state and the embedding table, scaled by a
single learned temperature. The log magnitudes cancel in the cosine, so the head only reads the embedding phase
angles, which it shares with the input embedding (`tie_word_embeddings=True`).

You can find the official checkpoints under the [MatterWave AI](https://huggingface.co/matterwaveai) organization.

## Usage example

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="matterwaveai/grt-30m",
    dtype=torch.float32,
    device=0,
)
pipe("Once upon a time", max_new_tokens=16)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("matterwaveai/grt-30m")
model = AutoModelForCausalLM.from_pretrained("matterwaveai/grt-30m", dtype=torch.float32)

inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=16)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Notes

- GRT is implemented in pure PyTorch and needs no custom kernel compilation. Optimized fused CUDA kernels are
  published separately and are not required to run the model.
- Every rotation acts on pairs of dimensions, so `head_dim`, `hidden_size` and `intermediate_size` must be even and
  split into whole butterfly blocks; [`GRTConfig`] raises a `ValueError` naming the offending width otherwise.
- [grt-30m](https://huggingface.co/matterwaveai/grt-30m) has 29.3M parameters: `hidden_size=256`, 12 blocks of 4
  heads, 11/21/1 rotation stages in the attention, feed-forward and mixer chains, and a 2048-token context. It uses the
  Qwen3 tokenizer of [Qwen/Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base); the embedding has
  151,936 rows, more than the tokenizer's vocabulary, as in Qwen3.
- The checkpoint is stored in fp32. Half precision works, with the position angles always computed in fp32
  and the learned frequencies kept in fp32, but the long rotation chains accumulate rounding error, so bf16 logits
  drift noticeably from fp32. Use fp32 when outputs must match the reference.

## GRTConfig

[[autodoc]] GRTConfig

## GRTModel

[[autodoc]] GRTModel
    - forward

## GRTForCausalLM

[[autodoc]] GRTForCausalLM
    - forward
