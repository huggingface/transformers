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
*This model was released on 2021-04-20 and added to Hugging Face Transformers on 2026-02-11.*


# NomicBERT

## Overview

The NomicBERT model currently has no academic papers specifically written about it, however, the [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) card clearly describes the model’s architecture and training approach: it extends BERT to a 2048 token context length, and modifies the BERT training procedure. Notable changes include: 

- Use [Rotary Position Embeddings](https://huggingface.co/papers/2104.09864.pdf) to allow for context length extrapolation.
- Use SiLU activations, which have [been shown](https://huggingface.co/papers/2002.05202) to [improve model performance](https://www.databricks.com/blog/mosaicbert)
- No dropout

> [!TIP]
> - NomicBERT can handle long sequences efficiently (up to 2048 tokens by default).
> - For masked language modeling, use `NomicBertForMaskedLM`.
> - Use smaller configs for testing locally to save memory and speed up unit tests.
> - Supports various heads: classification, QA, token classification, multiple choice, etc.


This model was contributed by community members ([Sonny Cooper](https://github.com/ed22699)).
The original code for nomic-embed-text-v1.5 can be found [here](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5).

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

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

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

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

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

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

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

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

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

## NomicBertForMultipleChoice

[[autodoc]] NomicBertForMultipleChoice

## NomicBertForNextSentencePrediction

[[autodoc]] NomicBertForNextSentencePrediction

## NomicBertForPreTraining

[[autodoc]] NomicBertForPreTraining

## NomicBertForQuestionAnswering

[[autodoc]] NomicBertForQuestionAnswering

## NomicBertForSequenceClassification

[[autodoc]] NomicBertForSequenceClassification

## NomicBertForTokenClassification

[[autodoc]] NomicBertForTokenClassification