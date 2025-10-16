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
*This model was released on 2024-05-31 and added to Hugging Face Transformers on 2024-08-06 and contributed by [Molbap](https://huggingface.co/Molbap).*

# Mamba 2

[Mamba2](https://huggingface.co/papers/2405.21060) introduce the State Space Duality (SSD) framework, which unifies aspects of attention mechanisms and SSMs. Using this framework, they design Mamba-2, an improved version of the Mamba architecture, with a refined selective SSM core layer. Mamba-2 achieves 2–8× faster performance than its predecessor while maintaining competitiveness with Transformers on language modeling tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="mistralai/Mamba-Codestral-7B-v0.1", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- Codestral Mamba has `groups=8` which are similar to the number of kv heads in an attention-based model.
- Codestral Mamba has two different forward passes: `torch_forward` or `cuda_kernels_forward`. Their results are expected to be slightly different.
- `torch_forward` without compilation is 3-4x faster than `cuda_kernels_forward`.
- `cuda_kernels_forward` uses the original CUDA kernels if they're available in your environment. It's slower during prefill because it requires a "warmup run" due to higher CPU overhead.
- This model has no positional embeddings, but it has an `attention_mask` and specific logic to mask out hidden states in two places during batched generation. This (and the reimplemented Mamba 2 kernels) results in a slight discrepancy between batched and cached generation.
- The SSM algorithm heavily relies on tensor contractions, which have matmul equivalents but the order of operations is slightly different. This makes the difference greater at smaller precisions.
- Hidden states corresponding to padding tokens are shutdown in 2 places and are mostly tested with left-padding. Right-padding propagates noise down the line and doesn't guarantee satisfactory results. Set `tokenizer.padding_side = "left"` to ensure you're using the correct padding side.

## Mamba2Config

[[autodoc]] Mamba2Config

## Mamba2Model

[[autodoc]] Mamba2Model
    - forward

## Mamba2LMHeadModel

[[autodoc]] Mamba2ForCausalLM
    - forward

