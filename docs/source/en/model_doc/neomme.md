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

*This model was contributed to Hugging Face Transformers on 2026-08-15.*

# NeoMME

[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg?style=for-the-badge)](https://arxiv.org)
[![Hugging Face](https://img.shields.io/badge/NeoMME_Collection-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Hcompany/neomme)

NeoMME is a family of bidirectional multilingual encoders for text and images, including natural images and documents. Text tokens and raw image patches use one shared Transformer encoder, without a separately pretrained vision encoder. The pretrained backbone returns contextual token representations for task-specific fine-tuning.

NeoMME-Retriever is a downstream model fine-tuned for text and document retrieval. It returns token embeddings for MaxSim scoring and mean-pooled embeddings for cosine similarity.

Details on NeoMME's training methodology, evaluations, benchmark performance, and ablations will be available in the technical report (coming soon), and open-source weights (Apache-2.0) can be found in the [NeoMME collection](https://huggingface.co/collections/Hcompany/neomme).

```python
import torch

from transformers import AutoModelForMaskedLM, AutoProcessor

checkpoint = "Hcompany/NeoMME-260M"
processor = AutoProcessor.from_pretrained(checkpoint)
model = AutoModelForMaskedLM.from_pretrained(checkpoint, device_map="auto")

text = f"The capital of {processor.tokenizer.mask_token} is London."
inputs = processor(text=[text], task="document").to(model.device)

with torch.inference_mode():
    outputs = model(**inputs)

masked_index = (inputs.input_ids[0] == processor.tokenizer.mask_token_id).nonzero().item()
predicted_token_id = outputs.logits[0, masked_index].argmax(dim=-1)
print(processor.tokenizer.decode(predicted_token_id))
```

```python
import requests
import torch
from PIL import Image

from transformers import NeoMMEForRetrieval, NeoMMEProcessor

checkpoint = "Hcompany/NeoMME-260M-Retriever"
processor = NeoMMEProcessor.from_pretrained(checkpoint)
model = NeoMMEForRetrieval.from_pretrained(checkpoint).eval()

# The document page screenshots from your corpus
image_urls = [
    "https://github.com/tonywu71/colpali-cookbooks/blob/main/examples/data/shift_kazakhstan.jpg?raw=true",
    "https://github.com/tonywu71/colpali-cookbooks/blob/main/examples/data/energy_electricity_generation.jpg?raw=true",
]
documents = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

# The queries you want to retrieve documents for
queries = [
    "Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?",
    "Which hour of the day had the highest overall electricity generation in 2019?",
]

inputs_documents = processor(images=documents).to(model.device)
inputs_text = processor(text=queries, task="query").to(model.device)

with torch.inference_mode():
    document_embeddings = model(**inputs_documents).embeddings
    query_embeddings = model(**inputs_text).embeddings

scores = processor.score_retrieval(query_embeddings, document_embeddings)
print(scores)  # Expected: scores[0, 0] > scores[0, 1] and scores[1, 1] > scores[1, 0].
```

## NeoMMEConfig

[[autodoc]] NeoMMEConfig

## NeoMMEImageProcessor

[[autodoc]] NeoMMEImageProcessor
    - preprocess

## NeoMMEProcessor

[[autodoc]] NeoMMEProcessor
    - __call__
    - process_queries
    - process_text_documents
    - process_images
    - score_retrieval

## NeoMMEModel

[[autodoc]] NeoMMEModel
    - forward

## NeoMMEForMaskedLM

[[autodoc]] NeoMMEForMaskedLM
    - forward

## NeoMMEForRetrieval

[[autodoc]] NeoMMEForRetrieval
    - forward
    - get_multivector_embeddings
    - get_dense_embeddings