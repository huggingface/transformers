---
language: "ca"
tags:
  - lm-head
  - masked-lm
  - catalan
  - exbert
license: mit
---

# Calbert: a Catalan Language Model

## Introduction

CALBERT is an open-source language model for Catalan pretrained on the ALBERT architecture.

It is now available on Hugging Face in its `tiny-uncased` version and `base-uncased` (the one you're looking at) as well, and was pretrained on the [OSCAR dataset](https://traces1.inria.fr/oscar/).

For further information or requests, please go to the [GitHub repository](https://github.com/codegram/calbert)

## Pre-trained models

| Model                               | Arch.          | Training data          |
| ----------------------------------- | -------------- | ---------------------- |
| `codegram` / `calbert-tiny-uncased` | Tiny (uncased) | OSCAR (4.3 GB of text) |
| `codegram` / `calbert-base-uncased` | Base (uncased) | OSCAR (4.3 GB of text) |

## How to use Calbert with HuggingFace

#### Load Calbert and its tokenizer:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("codegram/calbert-base-uncased")
model = AutoModel.from_pretrained("codegram/calbert-base-uncased")

model.eval() # disable dropout (or leave in train mode to finetune
```

#### Filling masks using pipeline

```python
from transformers import pipeline

calbert_fill_mask  = pipeline("fill-mask", model="codegram/calbert-base-uncased", tokenizer="codegram/calbert-base-uncased")
results = calbert_fill_mask("M'agrada [MASK] això")
# results
# [{'sequence': "[CLS] m'agrada molt aixo[SEP]", 'score': 0.614592969417572, 'token': 61},
#  {'sequence': "[CLS] m'agrada moltíssim aixo[SEP]", 'score': 0.06058056280016899, 'token': 4867},
#  {'sequence': "[CLS] m'agrada més aixo[SEP]", 'score': 0.017195818945765495, 'token': 43},
#  {'sequence': "[CLS] m'agrada llegir aixo[SEP]", 'score': 0.016321714967489243, 'token': 684},
#  {'sequence': "[CLS] m'agrada escriure aixo[SEP]", 'score': 0.012185849249362946, 'token': 1306}]

```

#### Extract contextual embedding features from Calbert output

```python
import torch
# Tokenize in sub-words with SentencePiece
tokenized_sentence = tokenizer.tokenize("M'és una mica igual")
# ['▁m', "'", 'es', '▁una', '▁mica', '▁igual']

# 1-hot encode and add special starting and end tokens
encoded_sentence = tokenizer.encode(tokenized_sentence)
# [2, 109, 7, 71, 36, 371, 1103, 3]
# NB: Can be done in one step : tokenize.encode("M'és una mica igual")

# Feed tokens to Calbert as a torch tensor (batch dim 1)
encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
embeddings, _ = model(encoded_sentence)
embeddings.size()
# torch.Size([1, 8, 768])
embeddings.detach()
# tensor([[[-0.0261,  0.1166, -0.1075,  ..., -0.0368,  0.0193,  0.0017],
#          [ 0.1289, -0.2252,  0.9881,  ..., -0.1353,  0.3534,  0.0734],
#          [-0.0328, -1.2364,  0.9466,  ...,  0.3455,  0.7010, -0.2085],
#          ...,
#          [ 0.0397, -1.0228, -0.2239,  ...,  0.2932,  0.1248,  0.0813],
#          [-0.0261,  0.1165, -0.1074,  ..., -0.0368,  0.0193,  0.0017],
#          [-0.1934, -0.2357, -0.2554,  ...,  0.1831,  0.6085,  0.1421]]])
```

## Authors

CALBERT was trained and evaluated by [Txus Bach](https://twitter.com/txustice), as part of [Codegram](https://www.codegram.com)'s applied research.

<a href="https://huggingface.co/exbert/?model=codegram/calbert-base-uncased&modelKind=bidirectional&sentence=M%27agradaria%20força%20saber-ne%20més">
	<img width="300px" src="https://cdn-media.huggingface.co/exbert/button.png">
</a>
