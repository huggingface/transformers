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
*This model was contributed to Hugging Face Transformers on 2026-06-29.*

# LFM2Bidirectional

## Overview

LFM2Bidirectional is the encoder (bidirectional) variant of [LFM2](./lfm2), used for retrieval and embedding
checkpoints such as [LiquidAI/LFM2.5-Embedding-350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M) and
[LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M).

## Architecture

It reuses the [LFM2](./lfm2) backbone (interleaved gated short-convolution and grouped-query attention blocks) with
two changes that turn the causal decoder into an encoder:

- **Bidirectional attention** — every token attends to all non-padding tokens (no causal mask), with
  `is_causal = False`.
- **Non-causal short convolution** — the short conv is centered instead of causal, so each position mixes both its
  left and right neighbors.

Both the Embedding and ColBERT checkpoints share this single architecture. The embedding pooling and the ColBERT
dense projection are handled by sentence-transformers and are not part of this model. The model is an encoder only:
it has no KV cache and does not support generation.

## Example

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "LiquidAI/LFM2.5-Embedding-350M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, dtype="bfloat16")

inputs = tokenizer(["A sentence to embed."], return_tensors="pt", padding=True)
with torch.no_grad():
    hidden_states = model(**inputs).last_hidden_state
```

## Lfm2BidirectionalConfig

[[autodoc]] Lfm2BidirectionalConfig

## Lfm2BidirectionalModel

[[autodoc]] Lfm2BidirectionalModel
    - forward

## Lfm2BidirectionalPreTrainedModel

[[autodoc]] Lfm2BidirectionalPreTrainedModel
    - forward
