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

[ColPali](https://huggingface.co/papers/2407.01449) is a model designed to retrieve documents by analyzing their visual features. Unlike traditional systems that rely heavily on text extraction and OCR, ColPali treats each page as an image. It uses [Paligemma-3B](./paligemma) to capture not only text, but also the layout, tables, charts, and other visual elements to create detailed embeddings. This offers a more comprehensive understanding of documents and enables more efficient and accurate retrieval.

You can find all the original ColPali checkpoints under the [ColPali](https://huggingface.co/collections/vidore/hf-native-colvision-models-6755d68fc60a8553acaa96f7) collection.

> [!TIP]
> Click on the ColPali models in the right sidebar for more examples of how to use ColPali for image retrieval.

<hfoptions id="usage">
<hfoption id="image retrieval">

```py
import requests
import torch
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor

# Load model (bfloat16 support is limited; fallback to float32 if needed)
model = ColPaliForRetrieval.from_pretrained(
    "vidore/colpali-v1.2-hf",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",  # "cpu", "cuda", or "mps" for Apple Silicon
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

url1 = "https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg"
url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg"

images = [
    Image.open(requests.get(url1, stream=True).raw),
    Image.open(requests.get(url2, stream=True).raw),
]

queries = [
    "Who printed the edition of Romeo and Juliet?",
    "When was the United States Declaration of Independence proclaimed?",
]

# Process the inputs
inputs_images = processor(images=images, return_tensors="pt").to(model.device)
inputs_text = processor(text=queries, return_tensors="pt").to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**inputs_images).embeddings
    query_embeddings = model(**inputs_text).embeddings

scores = processor.score_retrieval(query_embeddings, image_embeddings)

print("Retrieval scores (query x image):")
print(scores)
```
</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes.md) to quantize the weights to int4.

```py
import requests
import torch
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor
from transformers import BitsAndBytesConfig

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model_name = "vidore/colpali-v1.2-hf"

# Load model 
model = ColPaliForRetrieval.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda"
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

url1 = "https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg"
url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg"

images = [
    Image.open(requests.get(url1, stream=True).raw),
    Image.open(requests.get(url2, stream=True).raw),
]

queries = [
    "Who printed the edition of Romeo and Juliet?",
    "When was the United States Declaration of Independence proclaimed?",
]

# Process the inputs
inputs_images = processor(images=images, return_tensors="pt").to(model.device)
inputs_text = processor(text=queries, return_tensors="pt").to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**inputs_images).embeddings
    query_embeddings = model(**inputs_text).embeddings

scores = processor.score_retrieval(query_embeddings, image_embeddings)

print("Retrieval scores (query x image):")
print(scores)
```

## Notes

- [`~ColPaliProcessor.score_retrieval`] returns a 2D tensor where the first dimension is the number of queries and the second dimension is the number of images. A higher score indicates more similarity between the query and image.

## ColPaliConfig

[[autodoc]] ColPaliConfig

## ColPaliProcessor

[[autodoc]] ColPaliProcessor

## ColPaliForRetrieval

[[autodoc]] ColPaliForRetrieval
    - forward
