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
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base-oscar-4gb")
camembert = CamembertModel.from_pretrained("camembert/camembert-base-oscar-4gb")

camembert.eval()  # disable dropout (or leave in train mode to finetune)

```

##### Filling masks using pipeline 
```python
from transformers import pipeline 

camembert_fill_mask  = pipeline("fill-mask", model="camembert/camembert-base-oscar-4gb", tokenizer="camembert/camembert-base-oscar-4gb")
>>> results = camembert_fill_mask("Le camembert est <mask> !")
# results
#[{'sequence': '<s> Le camembert est parfait!</s>', 'score': 0.04089554399251938, 'token': 1654}, 
#{'sequence': '<s> Le camembert est délicieux!</s>', 'score': 0.037193264812231064, 'token': 7200}, 
#{'sequence': '<s> Le camembert est prêt!</s>', 'score': 0.025467922911047935, 'token': 1415}, 
#{'sequence': '<s> Le camembert est meilleur!</s>', 'score': 0.022812040522694588, 'token': 528},
#{'sequence': '<s> Le camembert est différent!</s>', 'score': 0.017135459929704666, 'token': 2935}]

```

##### Extract contextual embedding features from Camembert output 
```python
import torch
# Tokenize in sub-words with SentencePiece
tokenized_sentence = tokenizer.tokenize("J'aime le camembert !")
# ['▁J', "'", 'aime', '▁le', '▁ca', 'member', 't', '▁!'] 

# 1-hot encode and add special starting and end tokens 
encoded_sentence = tokenizer.encode(tokenized_sentence)
# [5, 121, 11, 660, 16, 730, 25543, 110, 83, 6]
# NB: Can be done in one step : tokenize.encode("J'aime le camembert !")

# Feed tokens to Camembert as a torch tensor (batch dim 1)
encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
embeddings, _ = camembert(encoded_sentence)
# embeddings.detach()
# embeddings.size torch.Size([1, 10, 768])
#tensor([[[-0.1120, -0.1464,  0.0181,  ..., -0.1723, -0.0278,  0.1606],
#         [ 0.1234,  0.1202, -0.0773,  ..., -0.0405, -0.0668, -0.0788],
#         [-0.0440,  0.0480, -0.1926,  ...,  0.1066, -0.0961,  0.0637],
#         ...,
```

##### Extract contextual embedding features from all Camembert layers
```python
from transformers import CamembertConfig
# (Need to reload the model with new config)
config = CamembertConfig.from_pretrained("camembert/camembert-base-oscar-4gb", output_hidden_states=True)
camembert = CamembertModel.from_pretrained("camembert/camembert-base-oscar-4gb", config=config)

embeddings, _, all_layer_embeddings = camembert(encoded_sentence)
#  all_layer_embeddings list of len(all_layer_embeddings) == 13 (input embedding layer + 12 self attention layers)
all_layer_embeddings[5]
# layer 5 contextual embedding : size torch.Size([1, 10, 768])
#tensor([[[-0.1584, -0.1207, -0.0179,  ...,  0.5457,  0.1491, -0.1191],
#         [-0.1122,  0.3634,  0.0676,  ...,  0.4395, -0.0470, -0.3781],
#         [-0.2232,  0.0019,  0.0140,  ...,  0.4461, -0.0233,  0.0735],
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
