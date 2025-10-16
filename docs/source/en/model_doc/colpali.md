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
*This model was released on 2024-06-27 and added to Hugging Face Transformers on 2024-12-17 and contributed by [tonywu71](https://huggingface.co/tonywu71) and [yonigozlan](https://huggingface.co/yonigozlan).*

# ColPali

[ColPali](https://huggingface.co/papers/2407.01449) is a retrieval model designed for visually rich documents that processes document pages as images rather than relying solely on text. It builds on recent vision-language models to generate high-quality contextualized embeddings that capture both textual and visual information. Using a late interaction matching mechanism, ColPali achieves faster and more accurate document retrieval compared to existing systems. The model is evaluated on the new Visual Document Retrieval Benchmark (ViDoRe), which spans diverse domains, languages, and retrieval settings.

<hfoptions id="usage">
<hfoption id="ColPaliForRetrieval">

```py
import requests
import torch
from PIL import Image
from transformers import ColPaliForRetrieval, AutoProcessor

model = ColPaliForRetrieval.from_pretrained("vidore/colpali-v1.3-hf",dtype="auto")
processor = AutoProcessor.from_pretrained("vidore/colpali-v1.3-hf")

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

</hfoption>
</hfoptions>

## Usage tips

- [`~ColPaliProcessor.score_retrieval`] returns a 2D tensor with dimensions `(number_of_queries, number_of_images)`. Higher scores indicate greater similarity between queries and images.

## ColPaliConfig

[[autodoc]] ColPaliConfig

## ColPaliProcessor

[[autodoc]] ColPaliProcessor

## ColPaliForRetrieval

[[autodoc]] ColPaliForRetrieval
    - forward
