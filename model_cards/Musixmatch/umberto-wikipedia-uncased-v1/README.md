---
language: italian
---

# UmBERTo Wikipedia Uncased

[UmBERTo](https://github.com/musixmatchresearch/umberto) is a Roberta-based Language Model trained on large Italian Corpora and uses two innovative approaches: SentencePiece and Whole Word Masking. Now available at [github.com/huggingface/transformers](https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1)

<p align="center">
    <img src="https://user-images.githubusercontent.com/7140210/72913702-d55a8480-3d3d-11ea-99fc-f2ef29af4e72.jpg" width="700"> </br>
    Marco Lodola, Monument to Umberto Eco, Alessandria 2019
</p>

## Dataset
UmBERTo-Wikipedia-Uncased Training is trained on a relative small corpus (~7GB) extracted from [Wikipedia-ITA](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/).

## Pre-trained model

| Model | WWM | Cased | Tokenizer | Vocab Size  | Train Steps |  Download |
| ------ | ------ | ------ | ------ | ------ |------ | ------ |
| `umberto-wikipedia-uncased-v1` | YES | YES | SPM | 32K | 100k | [Link](http://bit.ly/35wbSj6) |

This model was trained with [SentencePiece](https://github.com/google/sentencepiece) and Whole Word Masking.

## Downstream Tasks
These results refers to umberto-wikipedia-uncased model. All details are at [Umberto](https://github.com/musixmatchresearch/umberto) Official Page.

#### Named Entity Recognition (NER)

| Dataset | F1 | Precision | Recall | Accuracy |
| ------ | ------ | ------ |  ------ |  ----- |
| **ICAB-EvalITA07** | **86.240** | 85.939 | 86.544 | 98.534 | 
| **WikiNER-ITA** | **90.483** | 90.328 | 90.638 | 98.661 | 

#### Part of Speech (POS)

| Dataset | F1 | Precision | Recall | Accuracy |
| ------ | ------ | ------ |  ------ |  ------ |
| **UD_Italian-ISDT** | 98.563  | 98.508 | 98.618 | **98.717** | 
| **UD_Italian-ParTUT** | 97.810 | 97.835 |  97.784 | **98.060** | 



## Usage

##### Load UmBERTo Wikipedia Uncased with AutoModel, Autotokenizer:

```python

import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Musixmatch/umberto-wikipedia-uncased-v1")
umberto = AutoModel.from_pretrained("Musixmatch/umberto-wikipedia-uncased-v1")

encoded_input = tokenizer.encode("Umberto Eco è stato un grande scrittore")
input_ids = torch.tensor(encoded_input).unsqueeze(0)  # Batch size 1
outputs = umberto(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output
```

##### Predict masked token:

```python
from transformers import pipeline

fill_mask = pipeline(
	"fill-mask",
	model="Musixmatch/umberto-wikipedia-uncased-v1",
	tokenizer="Musixmatch/umberto-wikipedia-uncased-v1"
)

result = fill_mask("Umberto Eco è <mask> un grande scrittore")
# {'sequence': '<s> umberto eco è stato un grande scrittore</s>', 'score': 0.5784581303596497, 'token': 361}
# {'sequence': '<s> umberto eco è anche un grande scrittore</s>', 'score': 0.33813193440437317, 'token': 269}
# {'sequence': '<s> umberto eco è considerato un grande scrittore</s>', 'score': 0.027196012437343597, 'token': 3236}
# {'sequence': '<s> umberto eco è diventato un grande scrittore</s>', 'score': 0.013716378249228, 'token': 5742}
# {'sequence': '<s> umberto eco è inoltre un grande scrittore</s>', 'score': 0.010662357322871685, 'token': 1030}
```


## Citation
All of the original datasets are publicly available or were released with the owners' grant. The datasets are all released under a CC0 or CCBY license.

* UD Italian-ISDT Dataset [Github](https://github.com/UniversalDependencies/UD_Italian-ISDT)
* UD Italian-ParTUT Dataset [Github](https://github.com/UniversalDependencies/UD_Italian-ParTUT)
* I-CAB (Italian Content Annotation Bank), EvalITA [Page](http://www.evalita.it/)
* WIKINER [Page](https://figshare.com/articles/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500) , [Paper](https://www.sciencedirect.com/science/article/pii/S0004370212000276?via%3Dihub)

```
@inproceedings {magnini2006annotazione,
	title = {Annotazione di contenuti concettuali in un corpus italiano: I - CAB},
	author = {Magnini,Bernardo and Cappelli,Amedeo and Pianta,Emanuele and Speranza,Manuela and Bartalesi Lenzi,V and Sprugnoli,Rachele and Romano,Lorenza and Girardi,Christian and Negri,Matteo},
	booktitle = {Proc.of SILFI 2006},
	year = {2006}
}
@inproceedings {magnini2006cab,
	title = {I - CAB: the Italian Content Annotation Bank.},
	author = {Magnini,Bernardo and Pianta,Emanuele and Girardi,Christian and Negri,Matteo and Romano,Lorenza and Speranza,Manuela and Lenzi,Valentina Bartalesi and Sprugnoli,Rachele},
	booktitle = {LREC},
	pages = {963--968},
	year = {2006},
	organization = {Citeseer}
}
```

## Authors

**Loreto Parisi**: `loreto at musixmatch dot com`, [loretoparisi](https://github.com/loretoparisi)<br>
**Simone Francia**: `simone.francia at musixmatch dot com`, [simonefrancia](https://github.com/simonefrancia)<br>
**Paolo Magnani**: `paul.magnani95 at gmail dot com`, [paulthemagno](https://github.com/paulthemagno)<br>

## About Musixmatch AI
![Musxmatch Ai mac app icon-128](https://user-images.githubusercontent.com/163333/72244273-396aa380-35ee-11ea-894b-4ea48230c02b.png)<br>
We do Machine Learning and Artificial Intelligence @[musixmatch](https://twitter.com/Musixmatch)<br>
Follow us on [Twitter](https://twitter.com/musixmatchai) [Github](https://github.com/musixmatchresearch)

