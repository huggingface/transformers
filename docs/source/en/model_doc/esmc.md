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
*This model was contributed to Hugging Face Transformers on 2026-06-17.*

# ESMC

## Overview

ESMC (ESM Cambrian) is a family of protein language models released by [BioHub](https://biohub.org/).
It is a bidirectional Transformer encoder trained with a masked-language-modelling objective over amino-acid sequences.
Like [ESM-2](./esm), ESMC produces per-residue representations that are useful for downstream protein modelling tasks.

ESMC is suitable for fine-tuning on protein classification or token classification tasks. It is also used as the
backbone of [ESMFold2](./esmfold2), where it generates representations that are used as input to the folding head.

Pre-trained checkpoints are available on the Hugging Face Hub, including
[`biohub/ESMC-300M`](https://huggingface.co/biohub/ESMC-300M),
[`biohub/ESMC-600M`](https://huggingface.co/biohub/ESMC-600M) and
[`biohub/ESMC-6B`](https://huggingface.co/biohub/ESMC-6B).

## Usage example

```python
import torch
from transformers import AutoTokenizer, ESMCModel

tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-300M")
model = ESMCModel.from_pretrained("biohub/ESMC-300M")

inputs = tokenizer("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Per-residue representations of shape (batch, sequence_length, d_model).
representations = outputs.last_hidden_state
```

## ESMCConfig

[[autodoc]] ESMCConfig

## ESMCTokenizer

[[autodoc]] ESMCTokenizer

## ESMCModel

[[autodoc]] ESMCModel
    - forward

## ESMCForMaskedLM

[[autodoc]] ESMCForMaskedLM
    - forward

## ESMCForSequenceClassification

[[autodoc]] ESMCForSequenceClassification
    - forward

## ESMCForTokenClassification

[[autodoc]] ESMCForTokenClassification
    - forward
