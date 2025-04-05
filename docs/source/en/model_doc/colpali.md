<!--Copyright 2024 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# ColPali

[ColPali](https://arxiv.org/abs/2407.01449) is a model designed to retrieve documents by analyzing their visual features. Unlike traditional systems that rely heavily on text extraction and OCR, ColPali treats each page as an image, capturing not just the text but also the layout, tables, charts, and other visual elements. This approach allows it to understand documents more holistically. It leverages Vision Language Models (VLMs) to create detailed embeddings of these page images, enabling efficient and accurate retrieval. By integrating visual and textual data, ColPali offers a more comprehensive understanding of documents, making it particularly effective for complex documents where visual context is crucial.

You can find all Hf-native ColPali checkpoints under the [ColPali](https://huggingface.co/collections/vidore/hf-native-colvision-models-6755d68fc60a8553acaa96f7) collection.

> [!TIP]
> The orginal ColPali checkpoints are not natively supported by transformers 🤗. To use them you have to install [colpali-engine](https://github.com/illuin-tech/colpali). You can find the original checkpoints [here](https://huggingface.co/collections/vidore/colpali-models-673a5676abddf84949ce3180).

> [!TIP]
> Click on the ColPali models in the right sidebar for more examples of how to use ColPali for Image Retrieval.

<hfoptions id="usage">
<hfoption id="ImageRetrieval">

```py
import torch
from PIL import Image

from transformers import ColPaliForRetrieval, ColPaliProcessor

model_name = "vidore/colpali-v1.2-hf"

model = ColPaliForRetrieval.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

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
</hfoption>
</hfoptions>

## Notes

- The scores output by the `score_retrieval` method is a 2D tensor. First dimension is the number of queries and the second dimension is the number of images. The higher the score, the more similar the query and image are.

## ColPaliConfig

[[autodoc]] ColPaliConfig

## ColPaliProcessor

[[autodoc]] ColPaliProcessor

## ColPaliForRetrieval

[[autodoc]] ColPaliForRetrieval
    - forward
