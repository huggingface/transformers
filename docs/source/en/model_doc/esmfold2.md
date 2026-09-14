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
*This model was contributed to Hugging Face Transformers on 2026-08-19.*

# ESMFold2

## Overview

ESMFold2 is an all-atom protein structure prediction model. It predicts 3D coordinates and per-residue confidence
(pLDDT, PAE, PDE) directly from an amino-acid sequence, using the [ESMC](./esmc) protein language model as its
backbone. The architecture combines a sliding-window atom encoder with 3D rotary position embeddings, a pairwise
folding trunk applied iteratively, a diffusion-based structure head, and a confidence head.

The model checkpoint is available on the Hugging Face Hub at [`biohub/ESMFold2-hf`](https://huggingface.co/biohub/ESMFold2-hf).

## Usage example

```python
import torch

from transformers import EsmFold2Model

# The ESMC backbone is bundled in the checkpoint and loaded with the model.
# bf16 is the recommended inference precision.
model = EsmFold2Model.from_pretrained("biohub/ESMFold2-hf", dtype=torch.bfloat16, device_map="auto")

pdb_string = model.infer_protein_as_pdb("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")
print(pdb_string)
```

`infer_protein` returns the raw outputs (atom coordinates, distogram logits and confidence metrics) as an
[`~models.esmfold2.modeling_esmfold2.EsmFold2Output`] if you need them instead of a PDB string. You may get
slightly different predictions if you run the same sequence multiple times. Set a manual seed if you want exactly
reproducible structures.

ESMFold2 draws `config.structure_head.num_diffusion_samples` structures per fold. `infer_protein_as_pdb` renders the best-ranked
one (highest pTM); pass `sample_idx` to pick a specific sample instead. The PDB carries per-residue pLDDT in the
b-factor column, on the same 0-1 scale as the `plddt` output.

### `forward` vs `fold`

A structure prediction has two halves. `EsmFold2Model.forward` is the first: it runs the folding trunk over the
featurized inputs and returns the refined pair representation plus the distogram, as an
[`~models.esmfold2.modeling_esmfold2.EsmFold2TrunkOutput`]. It does not produce 3D coordinates — ESMFold2 gets those
by iterative denoising, and that sampling loop (the noise schedule, Kabsch alignment and the ODE/SDE update) lives in
`EsmFold2FoldingMixin` along with the confidence head call:

| Method | Use it for |
| --- | --- |
| `infer_protein_as_pdb(sequence)` | a PDB string, straight from an amino-acid sequence |
| `infer_protein(sequence)` | the raw [`~models.esmfold2.modeling_esmfold2.EsmFold2Output`] |
| `fold(**features)` | pre-featurized inputs (what `infer_protein` calls) |
| `forward(**features)` | the trunk alone — a distogram and pair representation, no sampling |

Call `fold` or `infer_protein` for an actual structure. Reach for `forward` when you only need the distogram, or when
you want to drive the diffusion sampler yourself: `fold` calls `forward` once and then hands its output to
`EsmFold2DiffusionModule`, whose own `forward` is the single denoising step.

## Faster inference with a fused kernel

The folding trunk's dominant cost is the triangle-multiplication update. Passing `use_kernels=True` to
[`~PreTrainedModel.from_pretrained`] swaps it for a fused Triton kernel loaded from the Hub via the
[`kernels`](https://github.com/huggingface/kernels) library, leaving the prediction unchanged. It is inference-only and
CUDA-only; on CPU or without the kernel installed the model transparently falls back to the pure-PyTorch implementation.
Make sure the model is on a CUDA device when kernelization happens (e.g. with `device_map`).

```python
import torch

from transformers import EsmFold2Model

model = EsmFold2Model.from_pretrained(
    "biohub/ESMFold2-hf", dtype=torch.bfloat16, device_map="cuda", use_kernels=True
)

pdb_string = model.infer_protein_as_pdb("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")
```

## EsmFold2Config

[[autodoc]] EsmFold2Config

## EsmFold2PreTrainedModel

[[autodoc]] EsmFold2PreTrainedModel

## EsmFold2Model

[[autodoc]] EsmFold2Model
    - forward
    - fold
    - infer_protein
    - infer_protein_as_pdb

## EsmFold2Output

[[autodoc]] models.esmfold2.modeling_esmfold2.EsmFold2Output

## EsmFold2TrunkOutput

[[autodoc]] models.esmfold2.modeling_esmfold2.EsmFold2TrunkOutput

## EsmFold2AtomInputs

[[autodoc]] models.esmfold2.modeling_esmfold2.EsmFold2AtomInputs
