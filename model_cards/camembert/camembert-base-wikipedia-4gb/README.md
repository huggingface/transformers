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
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base-wikipedia-4gb")
camembert = CamembertModel.from_pretrained("camembert/camembert-base-wikipedia-4gb")

camembert.eval()  # disable dropout (or leave in train mode to finetune)

```

##### Filling masks using pipeline 
```python
from transformers import pipeline 

camembert_fill_mask  = pipeline("fill-mask", model="camembert/camembert-base-wikipedia-4gb", tokenizer="camembert/camembert-base-wikipedia-4gb")
results = camembert_fill_mask("Le camembert est un fromage de <mask>!")
# results
#[{'sequence': '<s> Le camembert est un fromage de chèvre!</s>', 'score': 0.4937814474105835, 'token': 19370}, 
#{'sequence': '<s> Le camembert est un fromage de brebis!</s>', 'score': 0.06255942583084106, 'token': 30616}, 
#{'sequence': '<s> Le camembert est un fromage de montagne!</s>', 'score': 0.04340197145938873, 'token': 2364},
# {'sequence': '<s> Le camembert est un fromage de Noël!</s>', 'score': 0.02823255956172943, 'token': 3236}, 
#{'sequence': '<s> Le camembert est un fromage de vache!</s>', 'score': 0.021357402205467224, 'token': 12329}]
```

##### Extract contextual embedding features from Camembert output 
```python
import torch
# Tokenize in sub-words with SentencePiece
tokenized_sentence = tokenizer.tokenize("J'aime le camembert !")
# ['▁J', "'", 'aime', '▁le', '▁ca', 'member', 't', '▁!'] 

# 1-hot encode and add special starting and end tokens 
encoded_sentence = tokenizer.encode(tokenized_sentence)
# [5, 221, 10, 10600, 14, 8952, 10540, 75, 1114, 6]
# NB: Can be done in one step : tokenize.encode("J'aime le camembert !")

# Feed tokens to Camembert as a torch tensor (batch dim 1)
encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
embeddings, _ = camembert(encoded_sentence)
# embeddings.detach()
# embeddings.size torch.Size([1, 10, 768])
#tensor([[[-0.0928,  0.0506, -0.0094,  ..., -0.2388,  0.1177, -0.1302],
#         [ 0.0662,  0.1030, -0.2355,  ..., -0.4224, -0.0574, -0.2802],
#         [-0.0729,  0.0547,  0.0192,  ..., -0.1743,  0.0998, -0.2677],
#         ...,
```

##### Extract contextual embedding features from all Camembert layers
```python
from transformers import CamembertConfig
# (Need to reload the model with new config)
config = CamembertConfig.from_pretrained("camembert/camembert-base-wikipedia-4gb", output_hidden_states=True)
camembert = CamembertModel.from_pretrained("camembert/camembert-base-wikipedia-4gb", config=config)

embeddings, _, all_layer_embeddings = camembert(encoded_sentence)
#  all_layer_embeddings list of len(all_layer_embeddings) == 13 (input embedding layer + 12 self attention layers)
all_layer_embeddings[5]
# layer 5 contextual embedding : size torch.Size([1, 10, 768])
#tensor([[[-0.0059, -0.0227,  0.0065,  ..., -0.0770,  0.0369,  0.0095],
#         [ 0.2838, -0.1531, -0.3642,  ..., -0.0027, -0.8502, -0.7914],
#         [-0.0073, -0.0338, -0.0011,  ...,  0.0533, -0.0250, -0.0061],
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
