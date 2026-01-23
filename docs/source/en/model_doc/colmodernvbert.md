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
*This model was released on 2025-10-01 and added to Hugging Face Transformers on 2026-01-23.*

# ColModernVBert

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

ColModernVBert is a model for efficient visual document retrieval. It leverages [ModernVBert](modernvbert) to construct multi-vector embeddings directly from document images, following the ColPali approach.

The model was introduced in [ModernVBERT: Towards Smaller Visual Document Retrievers](https://huggingface.co/papers/2510.01149).

<hfoptions id="usage">
<hfoption id="Python">

```python
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import ColModernVBertProcessor, ColModernVBertForRetrieval

processor = ColModernVBertProcessor.from_pretrained("ModernVBERT/colmodernvbert-hf")
model = ColModernVBertForRetrieval.from_pretrained("ModernVBERT/colmodernvbert-hf")

# Load the test dataset
queries = [
    "A paint on the wall",
    "ColModernVBERT matches the performance of models nearly 10x larger on visual document benchmarks."
]

images = [
    Image.open(hf_hub_download("HuggingFaceTB/SmolVLM", "example_images/rococo.jpg", repo_type="space")),
    Image.open(hf_hub_download("ModernVBERT/colmodernvbert", "table.png", repo_type="model"))
]

# Preprocess the examples
batch_images = processor(images=images).to(model.device)
batch_queries = processor(text=queries).to(model.device)

# Run inference
with torch.inference_mode():
    image_embeddings = model(**batch_images).embeddings
    query_embeddings = model(**batch_queries).embeddings

# Compute retrieval scores
scores = processor.score_retrieval(
    query_embeddings=query_embeddings,
    passage_embeddings=image_embeddings,
)

scores = torch.softmax(scores, dim=-1)

print(scores)    # [[0.9350, 0.0650], [0.0015, 0.9985]]
```

</hfoption>
</hfoptions>

## ColModernVBertConfig

[[autodoc]] ColModernVBertConfig

## ColModernVBertProcessor

[[autodoc]] ColModernVBertProcessor

## ColModernVBertForRetrieval

[[autodoc]] ColModernVBertForRetrieval
    - forward
