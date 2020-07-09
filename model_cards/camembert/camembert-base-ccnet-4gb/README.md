---
language: french
---

# CamemBERT: a Tasty French Language Model

## Introduction

[CamemBERT](https://arxiv.org/abs/1911.03894) is a state-of-the-art language model for French based on the RoBERTa model. 

It is now available on Hugging Face in 6 different versions with varying number of parameters, amount of pretraining data and pretraining data source domains. 

For further information or requests, please go to [Camembert Website](https://camembert-model.fr/)

## Pre-trained models

| Model                          | #params                        | Arch. | Training data                     |
|--------------------------------|--------------------------------|-------|-----------------------------------|
| `camembert-base` | 110M   | Base  | OSCAR (138 GB of text)            |
| `camembert/camembert-large`              | 335M    | Large | CCNet (135 GB of text)            |
| `camembert/camembert-base-ccnet`         | 110M    | Base  | CCNet (135 GB of text)            |
| `camembert/camembert-base-wikipedia-4gb` | 110M    | Base  | Wikipedia (4 GB of text)          |
| `camembert/camembert-base-oscar-4gb`     | 110M    | Base  | Subsample of OSCAR (4 GB of text) |
| `camembert/camembert-base-ccnet-4gb`     | 110M    | Base  | Subsample of CCNet (4 GB of text) |

## How to use CamemBERT with HuggingFace

##### Load CamemBERT and its sub-word tokenizer :
```python
from transformers import CamembertModel, CamembertTokenizer

# You can replace "camembert-base" with any other model from the table, e.g. "camembert/camembert-large".
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base-ccnet-4gb")
camembert = CamembertModel.from_pretrained("camembert/camembert-base-ccnet-4gb")

camembert.eval()  # disable dropout (or leave in train mode to finetune)

```

##### Filling masks using pipeline 
```python
from transformers import pipeline 

camembert_fill_mask  = pipeline("fill-mask", model="camembert/camembert-base-ccnet-4gb", tokenizer="camembert/camembert-base-ccnet-4gb")
results = camembert_fill_mask("Le camembert est-il  <mask> ?")
# results
#[{'sequence': '<s> Le camembert est-il sain?</s>', 'score': 0.07001790404319763, 'token': 10286}, 
#{'sequence': '<s> Le camembert est-il français?</s>', 'score': 0.057594332844018936, 'token': 384}, 
#{'sequence': '<s> Le camembert est-il bon?</s>', 'score': 0.04098724573850632, 'token': 305}, 
#{'sequence': '<s> Le camembert est-il périmé?</s>', 'score': 0.03486393392086029, 'token': 30862}, 
#{'sequence': '<s> Le camembert est-il cher?</s>', 'score': 0.021535946056246758, 'token': 1604}]

```

##### Extract contextual embedding features from Camembert output 
```python
import torch
# Tokenize in sub-words with SentencePiece
tokenized_sentence = tokenizer.tokenize("J'aime le camembert !")
# ['▁J', "'", 'aime', '▁le', '▁ca', 'member', 't', '▁!'] 

# 1-hot encode and add special starting and end tokens 
encoded_sentence = tokenizer.encode(tokenized_sentence)
# [5, 133, 22, 1250, 16, 12034, 14324, 81, 76, 6]
# NB: Can be done in one step : tokenize.encode("J'aime le camembert !")

# Feed tokens to Camembert as a torch tensor (batch dim 1)
encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
embeddings, _ = camembert(encoded_sentence)
# embeddings.detach()
# embeddings.size torch.Size([1, 10, 768])
#tensor([[[ 0.0331,  0.0095, -0.2776,  ...,  0.2875, -0.0827, -0.2467],
#         [-0.1348,  0.0478, -0.5409,  ...,  0.8330,  0.0467,  0.0662],
#         [ 0.0920, -0.0264,  0.0177,  ...,  0.1112,  0.0108, -0.1123],
#         ...,
```

##### Extract contextual embedding features from all Camembert layers
```python
from transformers import CamembertConfig
# (Need to reload the model with new config)
config = CamembertConfig.from_pretrained("camembert/camembert-base-ccnet-4gb", output_hidden_states=True)
camembert = CamembertModel.from_pretrained("camembert/camembert-base-ccnet-4gb", config=config)

embeddings, _, all_layer_embeddings = camembert(encoded_sentence)
#  all_layer_embeddings list of len(all_layer_embeddings) == 13 (input embedding layer + 12 self attention layers)
all_layer_embeddings[5]
# layer 5 contextual embedding : size torch.Size([1, 10, 768])
#tensor([[[-0.0144,  0.1855,  0.4895,  ..., -0.1537,  0.0107, -0.2293],
#         [-0.6664, -0.0880, -0.1539,  ...,  0.3635,  0.4047,  0.1258],
#         [ 0.0511,  0.0540,  0.2545,  ...,  0.0709, -0.0288, -0.0779],
#         ...,
```


## Authors 

CamemBERT was trained and evaluated by Louis Martin\*, Benjamin Muller\*, Pedro Javier Ortiz Suárez\*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.


## Citation
If you use our work, please cite:

```bibtex
@inproceedings{martin2020camembert,
  title={CamemBERT: a Tasty French Language Model},
  author={Martin, Louis and Muller, Benjamin and Su{\'a}rez, Pedro Javier Ortiz and Dupont, Yoann and Romary, Laurent and de la Clergerie, {\'E}ric Villemonte and Seddah, Djam{\'e} and Sagot, Beno{\^\i}t},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```
