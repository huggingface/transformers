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

It is now available on Hugging Face in its `tiny-uncased` version (the one you're looking at) and `base-uncased` as well, and was pretrained on the [OSCAR dataset](https://traces1.inria.fr/oscar/).

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

tokenizer = AutoTokenizer.from_pretrained("codegram/calbert-tiny-uncased")
model = AutoModel.from_pretrained("codegram/calbert-tiny-uncased")

model.eval() # disable dropout (or leave in train mode to finetune
```

#### Filling masks using pipeline

```python
from transformers import pipeline

calbert_fill_mask  = pipeline("fill-mask", model="codegram/calbert-tiny-uncased", tokenizer="codegram/calbert-tiny-uncased")
results = calbert_fill_mask("M'agrada [MASK] això")
# results
# [{'sequence': "[CLS] m'agrada molt aixo[SEP]", 'score': 0.4403671622276306, 'token': 61},
#  {'sequence': "[CLS] m'agrada més aixo[SEP]", 'score': 0.050061386078596115, 'token': 43},
#  {'sequence': "[CLS] m'agrada veure aixo[SEP]", 'score': 0.026286985725164413, 'token': 157},
#  {'sequence': "[CLS] m'agrada bastant aixo[SEP]", 'score': 0.022483550012111664, 'token': 2143},
#  {'sequence': "[CLS] m'agrada moltíssim aixo[SEP]", 'score': 0.014491282403469086, 'token': 4867}]

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
# torch.Size([1, 8, 312])
embeddings.detach()
# tensor([[[-0.2726, -0.9855,  0.9643,  ...,  0.3511,  0.3499, -0.1984],
#         [-0.2824, -1.1693, -0.2365,  ..., -3.1866, -0.9386, -1.3718],
#         [-2.3645, -2.2477, -1.6985,  ..., -1.4606, -2.7294,  0.2495],
#         ...,
#         [ 0.8800, -0.0244, -3.0446,  ...,  0.5148, -3.0903,  1.1879],
#         [ 1.1300,  0.2425,  0.2162,  ..., -0.5722, -2.2004,  0.4045],
#         [ 0.4549, -0.2378, -0.2290,  ..., -2.1247, -2.2769, -0.0820]]])
```

## Authors

CALBERT was trained and evaluated by [Txus Bach](https://twitter.com/txustice), as part of [Codegram](https://www.codegram.com)'s applied research.

<a href="https://huggingface.co/exbert/?model=codegram/calbert-tiny-uncased&modelKind=bidirectional&sentence=M%27agradaria%20força%20saber-ne%20més">
	<img width="300px" src="https://hf-dinosaur.huggingface.co/exbert/button.png">
</a>
