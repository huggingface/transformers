---
language: italian
---

# UmBERTo Commoncrawl Cased

[UmBERTo](https://github.com/musixmatchresearch/umberto) is a Roberta-based Language Model trained on large Italian Corpora and uses two innovative approaches: SentencePiece and Whole Word Masking. Now available at [github.com/huggingface/transformers](https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1)

<p align="center">
    <img src="https://user-images.githubusercontent.com/7140210/72913702-d55a8480-3d3d-11ea-99fc-f2ef29af4e72.jpg" width="700"> </br>
    Marco Lodola, Monument to Umberto Eco, Alessandria 2019
</p>

## Dataset
UmBERTo-Commoncrawl-Cased utilizes the Italian subcorpus of [OSCAR](https://traces1.inria.fr/oscar/) as training set of the language model. We used deduplicated version of the Italian corpus that consists in 70 GB of plain text data, 210M sentences with 11B words where the sentences have been filtered and shuffled at line level in order to be used for NLP research.

## Pre-trained model

| Model | WWM | Cased | Tokenizer | Vocab Size  | Train Steps |  Download |
| ------ | ------ | ------ | ------ | ------ |------ | ------ |
| `umberto-commoncrawl-cased-v1` | YES | YES | SPM | 32K | 125k | [Link](http://bit.ly/35zO7GH) |

This model was trained with [SentencePiece](https://github.com/google/sentencepiece) and Whole Word Masking.

## Downstream Tasks
These results refers to umberto-commoncrawl-cased model. All details are at [Umberto](https://github.com/musixmatchresearch/umberto) Official Page.

#### Named Entity Recognition (NER)

| Dataset | F1 | Precision | Recall | Accuracy |
| ------ | ------ | ------ |  ------ |  ------ |
| **ICAB-EvalITA07** | **87.565**  | 86.596  | 88.556  | 98.690 | 
| **WikiNER-ITA** | **92.531**  | 92.509 | 92.553 | 99.136 | 

#### Part of Speech (POS)

| Dataset | F1 | Precision | Recall | Accuracy |
| ------ | ------ | ------ |  ------ |  ------ |
| **UD_Italian-ISDT** | 98.870  | 98.861 | 98.879 | **98.977** | 
| **UD_Italian-ParTUT** | 98.786 | 98.812 |  98.760 | **98.903** | 



## Usage

##### Load UmBERTo with AutoModel, Autotokenizer:

```python

import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Musixmatch/umberto-commoncrawl-cased-v1")
umberto = AutoModel.from_pretrained("Musixmatch/umberto-commoncrawl-cased-v1")

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
	model="Musixmatch/umberto-commoncrawl-cased-v1",
	tokenizer="Musixmatch/umberto-commoncrawl-cased-v1"
)

result = fill_mask("Umberto Eco è <mask> un grande scrittore")
# {'sequence': '<s> Umberto Eco è considerato un grande scrittore</s>', 'score': 0.18599839508533478, 'token': 5032}
# {'sequence': '<s> Umberto Eco è stato un grande scrittore</s>', 'score': 0.17816807329654694, 'token': 471}
# {'sequence': '<s> Umberto Eco è sicuramente un grande scrittore</s>', 'score': 0.16565583646297455, 'token': 2654}
# {'sequence': '<s> Umberto Eco è indubbiamente un grande scrittore</s>', 'score': 0.0932890921831131, 'token': 17908}
# {'sequence': '<s> Umberto Eco è certamente un grande scrittore</s>', 'score': 0.054701317101716995, 'token': 5269}
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

**Loreto Parisi**: `loreto at musixmatch dot com`, [loretoparisi](https://github.com/loretoparisi)
**Simone Francia**: `simone.francia at musixmatch dot com`, [simonefrancia](https://github.com/simonefrancia)
**Paolo Magnani**: `paul.magnani95 at gmail dot com`, [paulthemagno](https://github.com/paulthemagno)

## About Musixmatch AI
![Musxmatch Ai mac app icon-128](https://user-images.githubusercontent.com/163333/72244273-396aa380-35ee-11ea-894b-4ea48230c02b.png)
We do Machine Learning and Artificial Intelligence @[musixmatch](https://twitter.com/Musixmatch)
Follow us on [Twitter](https://twitter.com/musixmatchai) [Github](https://github.com/musixmatchresearch)


