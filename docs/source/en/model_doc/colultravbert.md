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
*This model was contributed to Hugging Face Transformers on 2026-10-16.*

# ColUltraVBert

<div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

ColUltraVBert is a late-interaction retriever for document page images. It applies the [ColModernVBert](colmodernvbert) recipe to [UltraVBert](ultravbert), an UltraBERT text encoder aligned with a SigLIP2 vision encoder. Every token of a page or a query becomes an L2-normalized multi-vector embedding, and [`ColUltraVBertProcessor.score_retrieval`] scores queries against pages by MaxSim.

[`ColUltraVBertProcessor`] tiles a page into 512-pixel tiles of 64 visual tokens each, plus a global image. The `max_image_tiles` setting of [`UltraVBertImageProcessor`] caps the tiles per page.

<hfoptions id="usage">
<hfoption id="Python">

```python
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import ColUltraVBertForRetrieval, ColUltraVBertProcessor


processor = ColUltraVBertProcessor.from_pretrained("athrael-soju/ColUltraVBERT")
model = ColUltraVBertForRetrieval.from_pretrained("athrael-soju/ColUltraVBERT", device_map="auto")

queries = [
    "A paint on the wall",
    "ColModernVBERT matches the performance of models nearly 10x larger on visual document benchmarks.",
]
images = [
    Image.open(hf_hub_download("HuggingFaceTB/SmolVLM", "example_images/rococo.jpg", repo_type="space")),
    Image.open(hf_hub_download("ModernVBERT/colmodernvbert", "table.png", repo_type="model")),
]

batch_images = processor.process_images(images=images).to(model.device)
batch_queries = processor.process_queries(text=queries).to(model.device)

with torch.inference_mode():
    image_embeddings = model(**batch_images).embeddings
    query_embeddings = model(**batch_queries).embeddings

scores = processor.score_retrieval(query_embeddings=query_embeddings, passage_embeddings=image_embeddings)
print(scores.softmax(dim=-1))
```

</hfoption>
</hfoptions>

## ColUltraVBertConfig

[[autodoc]] ColUltraVBertConfig

## ColUltraVBertProcessor

[[autodoc]] ColUltraVBertProcessor

## ColUltraVBertForRetrieval

[[autodoc]] ColUltraVBertForRetrieval
    - forward
