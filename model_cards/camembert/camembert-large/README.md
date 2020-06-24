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
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-large")
camembert = CamembertModel.from_pretrained("camembert/camembert-large")

camembert.eval()  # disable dropout (or leave in train mode to finetune)

```

##### Filling masks using pipeline 
```python
from transformers import pipeline 

camembert_fill_mask  = pipeline("fill-mask", model="camembert/camembert-large", tokenizer="camembert/camembert-large")
results = camembert_fill_mask("Le camembert est <mask> :)")
# results
#[{'sequence': '<s> Le camembert est bon :)</s>', 'score': 0.15560828149318695, 'token': 305}, 
#{'sequence': '<s> Le camembert est excellent :)</s>', 'score': 0.06821336597204208, 'token': 3497}, 
#{'sequence': '<s> Le camembert est délicieux :)</s>', 'score': 0.060438305139541626, 'token': 11661}, 
#{'sequence': '<s> Le camembert est ici :)</s>', 'score': 0.02023460529744625, 'token': 373}, 
#{'sequence': '<s> Le camembert est meilleur :)</s>', 'score': 0.01778135634958744, 'token': 876}]
```

##### Extract contextual embedding features from Camembert output 
```python
import torch
# Tokenize in sub-words with SentencePiece
tokenized_sentence = tokenizer.tokenize("J'aime le camembert !")
# ['▁J', "'", 'aime', '▁le', '▁cam', 'ember', 't', '▁!']

# 1-hot encode and add special starting and end tokens 
encoded_sentence = tokenizer.encode(tokenized_sentence)
# [5, 133, 22, 1250, 16, 12034, 14324, 81, 76, 6]
# NB: Can be done in one step : tokenize.encode("J'aime le camembert !")

# Feed tokens to Camembert as a torch tensor (batch dim 1)
encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
embeddings, _ = camembert(encoded_sentence)
# embeddings.detach()
# torch.Size([1, 10, 1024])
#tensor([[[-0.1284,  0.2643,  0.4374,  ...,  0.1627,  0.1308, -0.2305],
#         [ 0.4576, -0.6345, -0.2029,  ..., -0.1359, -0.2290, -0.6318],
#         [ 0.0381,  0.0429,  0.5111,  ..., -0.1177, -0.1913, -0.1121],
#         ...,
```

##### Extract contextual embedding features from all Camembert layers
```python
from transformers import CamembertConfig
# (Need to reload the model with new config)
config = CamembertConfig.from_pretrained("camembert/camembert-large", output_hidden_states=True)
camembert = CamembertModel.from_pretrained("camembert/camembert-large", config=config)

embeddings, _, all_layer_embeddings = camembert(encoded_sentence)
#  all_layer_embeddings list of len(all_layer_embeddings) == 25 (input embedding layer + 24 self attention layers)
all_layer_embeddings[5]
# layer 5 contextual embedding : size torch.Size([1, 10, 1024])
#tensor([[[-0.0600,  0.0742,  0.0332,  ..., -0.0525, -0.0637, -0.0287],
#         [ 0.0950,  0.2840,  0.1985,  ...,  0.2073, -0.2172, -0.6321],
#         [ 0.1381,  0.1872,  0.1614,  ..., -0.0339, -0.2530, -0.1182],
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
