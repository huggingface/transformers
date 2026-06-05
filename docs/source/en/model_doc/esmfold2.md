<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-06-04.*

# ESMFold2

## Overview

ESMFold2 is an all-atom protein structure prediction model. It predicts 3D coordinates and per-residue confidence
(pLDDT, PAE, PDE) directly from an amino-acid sequence, using the [ESMC](./esmc) protein language model as its
backbone. The architecture combines a sliding-window atom encoder with 3D rotary position embeddings, a pairwise
folding trunk applied iteratively (the "parcae" recurrence), a diffusion-based structure head, and a confidence head.

The ESMC backbone is **not bundled** in the ESMFold2 checkpoint; it is loaded separately from the repository named by
`config.esmc_id` (default [`biohub/ESMC-6B`](https://huggingface.co/biohub/ESMC-6B)).
[`ESMFold2Model.from_pretrained`](#transformers.ESMFold2Model) loads it automatically — pass `load_esmc=False` to skip
this (e.g. when supplying precomputed language-model hidden states).

The model checkpoint is available on the Hugging Face Hub at
[`biohub/ESMFold2`](https://huggingface.co/biohub/ESMFold2).

## Usage example

```python
import torch

from transformers import ESMFold2Model

# Loading the model also downloads and attaches the ESMC backbone (config.esmc_id).
# bf16 is the recommended inference precision.
model = ESMFold2Model.from_pretrained("biohub/ESMFold2", dtype=torch.bfloat16).cuda().eval()

pdb_string = model.infer_protein_as_pdb("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")
with open("prediction.pdb", "w") as f:
    f.write(pdb_string)
```

`infer_protein` returns the raw outputs (atom coordinates, distogram logits and confidence metrics) as a dictionary if
you need them instead of a PDB string. Generation is stochastic — set a manual seed for reproducible structures.

## Faster inference with a fused kernel

The folding trunk's dominant cost is the triangle-multiplication update. Passing `use_kernels=True` to
[`~PreTrainedModel.from_pretrained`] swaps it for a fused Triton kernel loaded from the Hub via the
[`kernels`](https://github.com/huggingface/kernels) library, leaving the prediction unchanged. It is inference-only and
CUDA-only; on CPU or without the kernel installed the model transparently falls back to the pure-PyTorch implementation.
Make sure the model is on a CUDA device when kernelization happens (e.g. with `device_map`).

```python
import torch

from transformers import ESMFold2Model

model = ESMFold2Model.from_pretrained(
    "biohub/ESMFold2", dtype=torch.bfloat16, device_map="cuda", use_kernels=True
).eval()

pdb_string = model.infer_protein_as_pdb("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")
```

## ESMFold2Config

[[autodoc]] ESMFold2Config

## ESMFold2Model

[[autodoc]] ESMFold2Model
    - forward
    - infer_protein
    - infer_protein_as_pdb
