<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-06-27 and added to Hugging Face Transformers on 2025-06-02 and contributed by [tonywu71](https://huggingface.co/tonywu71) and [yonigozlan](https://huggingface.co/yonigozlan).*

# ColQwen2

[ColQwen2](https://huggingface.co/papers/2407.01449) is a variant of the [ColPali](./colpali) model designed to retrieve documents by analyzing their visual features. Unlike traditional systems that rely heavily on text extraction and OCR, ColQwen2 treats each page as an image. It uses the [Qwen2-VL](./qwen2_vl) backbone to capture not only text, but also the layout, tables, charts, and other visual elements to create detailed multi-vector embeddings that can be used for retrieval by computing pairwise late interaction similarity scores. This offers a more comprehensive understanding of documents and enables more efficient and accurate retrieval.

<hfoptions id="usage">
<hfoption id="ColQwen2ForRetrieval">

```python
import requests
import torch
from PIL import Image
from transformers import ColQwen2ForRetrieval, AutoProcessor

model = ColQwen2ForRetrieval.from_pretrained("vidore/colqwen2-v1.0-hf",dtype="auto")
processor = AutoProcessor.from_pretrained("vidore/colqwen2-v1.0-hf")

url1 = "https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg"
url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg"

images = [
    Image.open(requests.get(url1, stream=True).raw),
    Image.open(requests.get(url2, stream=True).raw),
]

queries = [
    "When was the United States Declaration of Independence proclaimed?",
    "Who printed the edition of Romeo and Juliet?",
]

inputs_images = processor(images=images).to(model.device)
inputs_text = processor(text=queries).to(model.device)

with torch.no_grad():
    image_embeddings = model(**inputs_images).embeddings
    query_embeddings = model(**inputs_text).embeddings
scores = processor.score_retrieval(query_embeddings, image_embeddings)

print("Retrieval scores (query x image):")
print(scores)
```

You can also use checkpoints for `ColQwen2.5` that are **compatible with the ColQwen2 architecture**. This version of the model uses [Qwen2_5_VL](./qwen2_5_vl) as the backbone.

```python
import torch
from transformers import ColQwen2ForRetrieval, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available

model_name = "Sahil-Kabir/colqwen2.5-v0.2-hf" # An existing compatible checkpoint

model = ColQwen2ForRetrieval.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa"
)
processor = ColQwen2Processor.from_pretrained(model_name)
```

</hfoption>
</hfoptions>

## Usage tips

- [`~ColQwen2Processor.score_retrieval`] returns a 2D tensor with dimensions `(number_of_queries, number_of_images)`. Higher scores indicate greater similarity between queries and images.
- Unlike ColPali, ColQwen2 supports arbitrary image resolutions and aspect ratios. Images aren't resized into fixed-size squares, preserving more of the original input signal.
- Larger input images generate longer multi-vector embeddings. Adjust image resolution to balance performance and memory usage.

## ColQwen2Config

[[autodoc]] ColQwen2Config

## ColQwen2Processor

[[autodoc]] ColQwen2Processor

## ColQwen2ForRetrieval

[[autodoc]] ColQwen2ForRetrieval
    - forward
