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
*This model was contributed to Hugging Face Transformers on 2026-10-30.*

# H-Net

## Overview

[H-Net](https://huggingface.co/papers/2507.07955) (Dynamic Chunking for End-to-End Hierarchical Sequence Modeling) is a
byte-level language model from Cartesia that learns its own tokenization. There is no tokenizer: the input is raw UTF-8
bytes, and the model groups them into chunks with a learned routing module.

```text
bytes ─► encoder ─► routing ─► chunk ─► main network ─► dechunk ─► + residual ─► decoder ─► next byte
         (Mamba2)   (cos-sim    (keep     (Transformer,   (EMA over          (proj of
                     of next     boundary  or another      boundary           encoder
                     position)   positions) H-Net stage)   probabilities)     output)
```

A 1-stage model chunks once, a 2-stage model nests a second encoder/routing/decoder around the main network. Each
stage is wider than the previous one (`hidden_sizes`), and the input of a stage is padded with learned dimensions.

## Usage

The input ids are the UTF-8 bytes of the text, with `bos_token_id=254` prepended.

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("ArthurZ/hnet_1stage_L")

text = "Hello, my name is"
input_ids = torch.tensor([[model.config.bos_token_id, *text.encode()]])
output = model.generate(input_ids, max_new_tokens=50, do_sample=False)
print(bytes(output[0, 1:].tolist()).decode(errors="replace"))
```

The cache stores Mamba and attention states per stage, plus the last routing input and EMA value of each stage.
Cached decoding works with batch size 1. Each sequence closes chunks at different steps, so batched `generate` recomputes
the full sequence at every step.

## HNetConfig

[[autodoc]] HNetConfig

## HNetModel

[[autodoc]] HNetModel
    - forward

## HNetForCausalLM

[[autodoc]] HNetForCausalLM
    - forward
