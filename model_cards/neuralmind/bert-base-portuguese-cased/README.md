---
language: pt
license: mit
tags:
  - bert
  - pytorch
datasets:
  - brWaC
---

# BERTimbau Base (aka "bert-base-portuguese-cased")

![Bert holding a berimbau](https://imgur.com/JZ7Hynh.jpg)

## Introduction

BERTimbau Base is a pretrained BERT model for Brazilian Portuguese that achieves state-of-the-art performances on three downstream NLP tasks: Named Entity Recognition, Sentence Textual Similarity and Recognizing Textual Entailment. It is available in two sizes: Base and Large.

For further information or requests, please go to [BERTimbau repository](https://github.com/neuralmind-ai/portuguese-bert/).

## Available models

| Model                                    | Arch.      | #Layers | #Params |
| ---------------------------------------- | ---------- | ------- | ------- |
| `neuralmind/bert-base-portuguese-cased`  | BERT-Base  | 12      | 110M    |
| `neuralmind/bert-large-portuguese-cased` | BERT-Large | 24      | 335M    |

## Usage

```python
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPretraining  # Or BertForPreTraining for loading pretraining heads
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads

model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
```

### Masked language modeling prediction example

```python
from transformers import pipeline

pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer)

pipe('Tinha uma [MASK] no meio do caminho.')
# [{'score': 0.14287759363651276,
#  'sequence': '[CLS] Tinha uma pedra no meio do caminho. [SEP]',
#  'token': 5028,
#  'token_str': 'pedra'},
# {'score': 0.06213393807411194,
#  'sequence': '[CLS] Tinha uma árvore no meio do caminho. [SEP]',
#  'token': 7411,
#  'token_str': 'árvore'},
# {'score': 0.05515013635158539,
#  'sequence': '[CLS] Tinha uma estrada no meio do caminho. [SEP]',
#  'token': 5675,
#  'token_str': 'estrada'},
# {'score': 0.0299188531935215,
#  'sequence': '[CLS] Tinha uma casa no meio do caminho. [SEP]',
#  'token': 1105,
#  'token_str': 'casa'},
# {'score': 0.025660505518317223,
#  'sequence': '[CLS] Tinha uma cruz no meio do caminho. [SEP]',
#  'token': 3466,
#  'token_str': 'cruz'}]

```

### For BERT embeddings

```python
import torch

model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
input_ids = tokenizer.encode('Tinha uma pedra no meio do caminho.', return_tensors='pt')

with torch.no_grad():
    outs = model(input_ids)
    encoded = outs[0][0, 1:-1]  # Ignore [CLS] and [SEP] special tokens

# encoded.shape: (8, 768)
# tensor([[-0.0398, -0.3057,  0.2431,  ..., -0.5420,  0.1857, -0.5775],
#         [-0.2926, -0.1957,  0.7020,  ..., -0.2843,  0.0530, -0.4304],
#         [ 0.2463, -0.1467,  0.5496,  ...,  0.3781, -0.2325, -0.5469],
#         ...,
#         [ 0.0662,  0.7817,  0.3486,  ..., -0.4131, -0.2852, -0.2819],
#         [ 0.0662,  0.2845,  0.1871,  ..., -0.2542, -0.2933, -0.0661],
#         [ 0.2761, -0.1657,  0.3288,  ..., -0.2102,  0.0029, -0.2009]])
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
