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

*This model was contributed to Hugging Face Transformers on 2026-08-20.*

# NeoMME

[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg?style=for-the-badge)](https://arxiv.org)
[![Hugging Face](https://img.shields.io/badge/NeoMME_Collection-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Hcompany/neomme)

NeoMME is a family of efficient 260M and 800M parameter multimodal-native multilingual foundation encoders from H Company. It processes multilingual text tokens and raw image patches in a single bidirectional Transformer encoder, without a separately pretrained vision tower or causal language model.

NeoMME-Retriever is a model fine-tuned from the NeoMME backbone for visual document retrieval with joint late-interaction and dense objectives. It produces multivector embeddings for MeanMaxSim scoring (late-interaction) and mean-pooled embeddings for cosine similarity (dense).

The pretrained backbones and retrieval checkpoints are available under Apache 2.0 in the [NeoMME collection](https://huggingface.co/collections/Hcompany/neomme) and can be used with [Sentence Transformers](https://huggingface.co/sentence-transformers).

```python
import torch

from transformers import AutoModelForMaskedLM, AutoProcessor


model_name = "Hcompany/NeoMME-260M"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name, device_map="auto")

text = f"The capital of {processor.tokenizer.mask_token} is London."
inputs = processor(text=[text], return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model(**inputs)

masked_index = (inputs.input_ids[0] == processor.tokenizer.mask_token_id).nonzero().item()
predicted_token_id = outputs.logits[0, masked_index].argmax(dim=-1)
print(processor.tokenizer.decode(predicted_token_id))
```

> [!IMPORTANT]
> Install `sentence-transformers>=6.0.0` to use MeanMaxSim scoring in the retrieval example below. For the
> Sentence Transformers API, see the [Multi-Vector Encoder quickstart](https://sbert.net/docs/quickstart.html#multi-vector-encoder).

```python
from typing import Any, Literal

import requests
import torch
from PIL import Image
from sentence_transformers.util import mean_maxsim

from transformers import BatchFeature, NeoMMEForRetrieval, NeoMMEProcessor


def encode(
    messages: list[list[dict[str, Any]]],
    task: Literal["query", "document"],
) -> BatchFeature:
    return processor.apply_chat_template(
        messages,
        task=task,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": "longest"},
    )


model_name = "Hcompany/NeoMME-260M-Retriever"
processor = NeoMMEProcessor.from_pretrained(model_name)
model = NeoMMEForRetrieval.from_pretrained(model_name)

# Document images (our corpus)
image_urls = [
    "https://github.com/tonywu71/colpali-cookbooks/blob/6ef1332da6bcb48c7ef1f19b25bfa555be7031a8/examples/data/shift_kazakhstan.jpg?raw=true",
    "https://github.com/tonywu71/colpali-cookbooks/blob/6ef1332da6bcb48c7ef1f19b25bfa555be7031a8/examples/data/energy_electricity_generation.jpg?raw=true",
]
documents = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

# Queries
queries = [
    "Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?",
    "Which hour of the day had the highest overall electricity generation in 2019?",
]

document_messages = [
    [{"role": "user", "content": [{"type": "image", "image": document}]}] for document in documents
]
query_messages = [[{"role": "user", "content": query}] for query in queries]

inputs_documents = encode(document_messages, "document").to(model.device)
inputs_text = encode(query_messages, "query").to(model.device)

with torch.inference_mode():
    document_embeddings = model(**inputs_documents).embeddings
    query_embeddings = model(**inputs_text).embeddings

scores = mean_maxsim(
    query_embeddings,
    document_embeddings,
    a_mask=inputs_text["attention_mask"],
    b_mask=inputs_documents["attention_mask"],
)
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
    - apply_chat_template

## NeoMMEModel

[[autodoc]] NeoMMEModel
    - forward

## NeoMMEForMaskedLM

[[autodoc]] NeoMMEForMaskedLM
    - forward

## NeoMMEForRetrieval

[[autodoc]] NeoMMEForRetrieval
    - forward