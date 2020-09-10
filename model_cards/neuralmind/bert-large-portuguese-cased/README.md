---
language: pt
license: mit
tags:
  - bert
  - pytorch
datasets:
  - brWaC
---

# BERTimbau Large (aka "bert-large-portuguese-cased")

![Bert holding a berimbau](https://imgur.com/JZ7Hynh.jpg)

## Introduction

BERTimbau Large is a pretrained BERT model for Brazilian Portuguese that achieves state-of-the-art performances on three downstream NLP tasks: Named Entity Recognition, Sentence Textual Similarity and Recognizing Textual Entailment. It is available in two sizes: Base and Large.

For further information or requests, please go to [BERTimbau repository](https://github.com/neuralmind-ai/portuguese-bert/).

## Available models

| Model                                    | Arch.      | #Layers | #Params |
| ---------------------------------------- | ---------- | ------- | ------- |
| `neuralmind/bert-base-portuguese-cased`  | BERT-Base  | 12      | 110M    |
| `neuralmind/bert-large-portuguese-cased` | BERT-Large | 24      | 335M    |

## Usage

```python
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads

model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-large-portuguese-cased')
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased', do_lower_case=False)
```

### Masked language modeling prediction example

```python
from transformers import pipeline

pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer)

pipe('Tinha uma [MASK] no meio do caminho.')
# [{'score': 0.5054386258125305,
#   'sequence': '[CLS] Tinha uma pedra no meio do caminho. [SEP]',
#   'token': 5028,
#   'token_str': 'pedra'},
#  {'score': 0.05616172030568123,
#   'sequence': '[CLS] Tinha uma curva no meio do caminho. [SEP]',
#   'token': 9562,
#   'token_str': 'curva'},
#  {'score': 0.02348282001912594,
#   'sequence': '[CLS] Tinha uma parada no meio do caminho. [SEP]',
#   'token': 6655,
#   'token_str': 'parada'},
#  {'score': 0.01795753836631775,
#   'sequence': '[CLS] Tinha uma mulher no meio do caminho. [SEP]',
#   'token': 2606,
#   'token_str': 'mulher'},
#  {'score': 0.015246033668518066,
#   'sequence': '[CLS] Tinha uma luz no meio do caminho. [SEP]',
#   'token': 3377,
#   'token_str': 'luz'}]

```

### For BERT embeddings

```python

import torch

model = AutoModel.from_pretrained('neuralmind/bert-large-portuguese-cased')
input_ids = tokenizer.encode('Tinha uma pedra no meio do caminho.', return_tensors='pt')

with torch.no_grad():
    outs = model(input_ids)
    encoded = outs[0][0, 1:-1]  # Ignore [CLS] and [SEP] special tokens

# encoded.shape: (8, 1024)
# tensor([[ 1.1872,  0.5606, -0.2264,  ...,  0.0117, -0.1618, -0.2286],
#         [ 1.3562,  0.1026,  0.1732,  ..., -0.3855, -0.0832, -0.1052],
#         [ 0.2988,  0.2528,  0.4431,  ...,  0.2684, -0.5584,  0.6524],
#         ...,
#         [ 0.3405, -0.0140, -0.0748,  ...,  0.6649, -0.8983,  0.5802],
#         [ 0.1011,  0.8782,  0.1545,  ..., -0.1768, -0.8880, -0.1095],
#         [ 0.7912,  0.9637, -0.3859,  ...,  0.2050, -0.1350,  0.0432]])
```

## Citation

If you use our work, please cite:

```bibtex
@inproceedings{souza2020bertimbau,
  author    = {F{\'a}bio Souza and
               Rodrigo Nogueira and
               Roberto Lotufo},
  title     = {{BERT}imbau: pretrained {BERT} models for {B}razilian {P}ortuguese},
  booktitle = {9th Brazilian Conference on Intelligent Systems, {BRACIS}, Rio Grande do Sul, Brazil, October 20-23 (to appear)},
  year      = {2020}
}
```
