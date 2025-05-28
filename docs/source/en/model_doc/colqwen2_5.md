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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# ColQwen2.5

[ColQwen2](https://doi.org/10.48550/arXiv.2407.01449) is a variant of the [ColPali](./colpali) model designed to retrieve documents by analyzing their visual features. Unlike traditional systems that rely heavily on text extraction and OCR, ColQwen2.5 treats each page as an image. It uses the [Qwen2.5-VL](./qwen2_5_vl) backbone to capture not only text, but also the layout, tables, charts, and other visual elements to create detailed multi-vector embeddings that can be used for retrieval by computing pairwise late interaction similarity scores. This offers a more comprehensive understanding of documents and enables more efficient and accurate retrieval.

This model was contributed by [@tonywu71](https://huggingface.co/tonywu71) (ILLUIN Technology), [@yonigozlan](https://huggingface.co/yonigozlan) (HuggingFace) and [@qnguyen3](https://huggingface.co/qnguyen3) (WARA Media & Language).

You can find all the original ColPali checkpoints under Vidore's [Hf-native ColVision Models](https://huggingface.co/collections/vidore/hf-native-colvision-models-6755d68fc60a8553acaa96f7) collection.

> [!TIP]
> Click on the ColQwen2.5 models in the right sidebar for more examples of how to use ColQwen2.5 for image retrieval.

<hfoptions id="usage">
<hfoption id="image retrieval">

```python
import requests
import torch
from PIL import Image

from transformers import BitsAndBytesConfig, ColQwen2_5ForRetrieval, ColQwen2_5Processor


model_name = "qnguyen3/colqwen2_5-v0.2-hf"

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = ColQwen2_5ForRetrieval.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda",
).eval()

processor = ColQwen2_5Processor.from_pretrained(model_name)

url1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url2 = "http://images.cocodataset.org/val2017/000000212573.jpg"

images = [
    Image.open(requests.get(url1, stream=True).raw),
    Image.open(requests.get(url2, stream=True).raw),
]

queries = [
    "WHat are the colors of the two cats?",
    "Who printed the edition of Romeo and Juliet?",
]

# Process the inputs
inputs_images = processor(images=images, return_tensors="pt").to(model.device)
inputs_text = processor(text=queries, return_tensors="pt").to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**inputs_images).embeddings
    query_embeddings = model(**inputs_text).embeddings

# Score the queries against the images
scores = processor.score_retrieval(query_embeddings, image_embeddings)

print("Retrieval scores (query x image):")
print(scores)
```

## Notes

- [`~ColQwen2_5Processor.score_retrieval`] returns a 2D tensor where the first dimension is the number of queries and the second dimension is the number of images. A higher score indicates more similarity between the query and image.
- Unlike ColPali, ColQwen2.5 supports arbitrary image resolutions and aspect ratios, which means images are not resized into fixed-size squares. This preserves more of the original input signal.
- Larger input images generate longer multi-vector embeddings, allowing users to adjust image resolution to balance performance and memory usage.

## ColQwen2_5Config

[[autodoc]] ColQwen2_5Config

## ColQwen2_5Processor

[[autodoc]] ColQwen2_5Processor

## ColQwen2_5ForRetrieval

[[autodoc]] ColQwen2_5ForRetrieval
    - forward