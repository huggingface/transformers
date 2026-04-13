<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was released on 2024-02-10 and added to Hugging Face Transformers on 2026-04-01.*


# NomicBERT

## Overview

NomicBERT was proposed in [Nomic Embed: Training a Reproducible Long Context Text Embedder](https://huggingface.co/papers/2402.01613) by 
Zach Nussbaum, John X. Morris, Brandon Duderstadt, and Andriy Mulyar. It is BERT-inspired with the most notable extension applying 
[Rotary Position Embeddings](https://huggingface.co/papers/2104.09864.pdf) to an encoder model. 

The abstract from the paper is the following:

*This technical report describes the training of nomic-embed-text-v1, the first fully reproducible, open-source, open-weights, open-data, 8192 context length English text embedding model that outperforms both OpenAI Ada-002 and OpenAI text-embedding-3-small on the short-context MTEB benchmark and the long context LoCo benchmark. We release the training code and model weights under an Apache 2.0 license. In contrast with other open-source models, we release the full curated training data and code that allows for full replication of nomic-embed-text-v1. [...]*

This model was contributed by community member ([Sonny Cooper](https://github.com/ed22699)).
The original code for nomic-embed-text-v1.5 and nomic-embed-text-v1 can be found [here](https://github.com/nomic-ai/contrastors).

## Usage examples
The examples below demonstrate how to generate dense vector embeddings for different tasks using `[AutoModel]`. Each task requires a specific instruction prefix to optimize the embedding space for that use case.

<hfoptions id="usage">
<hfoption id="Search Document">

```py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

model_id = "nomic-ai/nomic-embed-text-v1.5"
revision = "refs/pr/57"

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModel.from_pretrained(model_id, revision=revision)

sentences = ['search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten']
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded_input)

embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
embeddings = F.normalize(embeddings, p=2, dim=1)
print(embeddings)
```

</hfoption>
<hfoption id="Search Query">

```py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

model_id = "nomic-ai/nomic-embed-text-v1.5"
revision = "refs/pr/57"

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModel.from_pretrained(model_id, revision=revision)

sentences = ['search_query: Who is Laurens van Der Maaten?']
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded_input)

embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
embeddings = F.normalize(embeddings, p=2, dim=1)
print(embeddings)
```

</hfoption>
<hfoption id="Clustering">

```py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

model_id = "nomic-ai/nomic-embed-text-v1.5"
revision = "refs/pr/57"

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModel.from_pretrained(model_id, revision=revision)

sentences = ['clustering: the quick brown fox']
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded_input)

embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
embeddings = F.normalize(embeddings, p=2, dim=1)
print(embeddings)
```

</hfoption>
<hfoption id="Classification">

```py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

model_id = "nomic-ai/nomic-embed-text-v1.5"
revision = "refs/pr/57"

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModel.from_pretrained(model_id, revision=revision)

sentences = ['classification: the quick brown fox']
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded_input)

embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
embeddings = F.normalize(embeddings, p=2, dim=1)
print(embeddings)
```

</hfoption>
</hfoptions>

## Extending the base context length
You can also increase the context length of the base model by giving dynamic rope parameters:

```python

model_id = "nomic-ai/nomic-embed-text-v1.5"
revision = "refs/pr/57"

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, model_max_length=8192)

# dynamic RoPE for increased context
rope_parameters = {"rope_theta": 1000.0, "rope_type": "dynamic", "factor": 2.0}
model = AutoModel.from_pretrained(model_id, revision=revision, rope_parameters=rope_parameters) 
```

## Notes

- NomicBERT uses Rotary Positional Embeddings (RoPE). For correct positional encoding either use 
    - right padding (default)
    - left padding and prepare `position_ids` accordingly

## NomicBertConfig

[[autodoc]] NomicBertConfig

## NomicBertModel

[[autodoc]] NomicBertModel
    - forward

## NomicBertForMaskedLM

[[autodoc]] NomicBertForMaskedLM

## NomicBertForSequenceClassification

[[autodoc]] NomicBertForSequenceClassification

## NomicBertForTokenClassification

[[autodoc]] NomicBertForTokenClassification
    - forward
