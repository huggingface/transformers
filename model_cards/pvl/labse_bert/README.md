---
language: en
thumbnail:
tags:
- bert
- embeddings
license: Apache-2.0
---

# LABSE BERT

## Model description

Model for "Language-agnostic BERT Sentence Embedding" paper from Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, Wei Wang. Model available in [TensorFlow Hub](https://tfhub.dev/google/LaBSE/1).

## Intended uses & limitations

#### How to use

```python
from transformers import AutoTokenizer, AutoModel
import torch

# from sentence-transformers
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

tokenizer = AutoTokenizer.from_pretrained("pvl/labse_bert", do_lower_case=False)
model = AutoModel.from_pretrained("pvl/labse_bert")

sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']

encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded_input)

sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])


```
