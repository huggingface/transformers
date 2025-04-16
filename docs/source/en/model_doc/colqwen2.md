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

# ColQwen2

## Overview

*ColQwen2* is a variant of the *ColPali* model, first introduced in [ColPali: Efficient Document Retrieval with Vision Language Models](https://doi.org/10.48550/arXiv.2407.01449) by **Manuel Faysse***, **Hugues Sibille***, **Tony Wu***, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo (* denotes equal contribution). Work led by ILLUIN Technology.

*ColQwen2* leverages a Vision Language Model (VLM) to construct efficient multi-vector embeddings directly from document images (“screenshots”) for document retrieval. We train the model to maximize the similarity between these document embeddings and the corresponding query embeddings, using the late interaction method introduced in ColBERT.

Using *ColQwen2* removes the need for potentially complex and brittle layout recognition and OCR pipelines with a single model that can take into account both the textual and visual content (layout, charts, etc.) of a document.

Unlike ColPali, ColQwen2—powered by the Qwen2-VL backbone—supports arbitrary image resolutions and aspect ratios. This makes it particularly interesting for document processing: images are not resized into fixed-size squares, preserving more of the original input signal. Larger input images generate longer multi-vector embeddings, allowing users to adjust image resolution to balance performance and memory usage.

## Resources

- The *ColPali* arXiv paper can be found [here](https://doi.org/10.48550/arXiv.2407.01449). 📄
- The official blog post detailing ColPali can be found [here](https://huggingface.co/blog/manu/colpali). 📝
- The ColPali Hf model card can be found [here](https://github.com/huggingface/transformers/tree/main/docs/source/en/model_doc/colpali.md). 🤗
- The original model implementation code for the ColQwen2 model and for the `colpali-engine` package can be found [here](https://github.com/illuin-tech/colpali). 🌎
- Cookbooks for learning to use the transformers-native version of *ColPali*, fine-tuning, and similarity maps generation can be found [here](https://github.com/tonywu71/colpali-cookbooks). 📚

This model was contributed by [@tonywu71](https://huggingface.co/tonywu71) and [@yonigozlan](https://huggingface.co/yonigozlan).

## Usage

This example demonstrates how to use *ColQwen2* to embed both queries and images, calculate their similarity scores, and identify the most relevant matches. For a specific query, you can retrieve the top-k most similar images by selecting the ones with the highest similarity scores.

```python
import torch
from PIL import Image

from transformers import ColQwen2ForRetrieval, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available

model_name = "vidore/colqwen2-v1.0-hf"

model = ColQwen2ForRetrieval.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2Processor.from_pretrained(model_name)

# Your inputs (replace dummy images with screenshots of your documents)
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]
queries = [
    "What is the organizational structure for our R&D department?",
    "Can you provide a breakdown of last year’s financial performance?",
]

# Process the inputs
batch_images = processor(images=images).to(model.device)
batch_queries = processor(text=queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images).embeddings
    query_embeddings = model(**batch_queries).embeddings

# Score the queries against the images
scores = processor.score_retrieval(query_embeddings, image_embeddings)
```

## ColQwen2Config

[[autodoc]] ColQwen2Config

## ColQwen2Processor

[[autodoc]] ColQwen2Processor

## ColQwen2ForRetrieval

[[autodoc]] ColQwen2ForRetrieval
    - forward
