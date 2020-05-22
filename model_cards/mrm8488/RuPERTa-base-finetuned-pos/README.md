---
language: spanish
thumbnail:
---

# RuPERTa-base  (Spanish RoBERTa) + POS 🎃🏷

This model is a fine-tuned on [CONLL CORPORA](https://www.kaggle.com/nltkdata/conll-corpora) version of [RuPERTa-base](https://huggingface.co/mrm8488/RuPERTa-base) for **POS** downstream task.

## Details of the downstream task (POS) - Dataset

- [Dataset:  CONLL Corpora ES](https://www.kaggle.com/nltkdata/conll-corpora) 📚

| Dataset                | # Examples |
| ---------------------- | ----- |
| Train                  | 445 K |
| Dev                    | 55 K |

- [Fine-tune on NER script provided by Huggingface](https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py)

- Labels covered:

```
ADJ
ADP
ADV
AUX
CCONJ
DET
INTJ
NOUN
NUM
PART
PRON
PROPN
PUNCT
SCONJ
SYM
VERB
```

## Metrics on evaluation set 🧾

|                                                      Metric                                                       |  # score  |
| :------------------------------------------------------------------------------------: | :-------: |
| F1                                       | **97.39**  
| Precision                                | **97.47** | 
| Recall                                   | **9732** |    

## Model in action 🔨


Example of usage

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('mrm8488/RuPERTa-base-finetuned-pos')
model = AutoModelForTokenClassification.from_pretrained('mrm8488/RuPERTa-base-finetuned-pos')

id2label = {
    "0": "O",
    "1": "ADJ",
    "2": "ADP",
    "3": "ADV",
    "4": "AUX",
    "5": "CCONJ",
    "6": "DET",
    "7": "INTJ",
    "8": "NOUN",
    "9": "NUM",
    "10": "PART",
    "11": "PRON",
    "12": "PROPN",
    "13": "PUNCT",
    "14": "SCONJ",
    "15": "SYM",
    "16": "VERB"
}

text ="Mis amigos están pensando viajar a Londres este verano."
input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

outputs = model(input_ids)
last_hidden_states = outputs[0]

for m in last_hidden_states:
  for index, n in enumerate(m):
    if(index > 0 and index <= len(text.split(" "))):
      print(text.split(" ")[index-1] + ": " + id2label[str(torch.argmax(n).item())])
      
'''
Output:
--------
Mis: NUM
amigos: PRON
están: AUX
pensando: ADV
viajar: VERB
a: ADP
Londres: PROPN
este: DET
verano..: NOUN
'''
```
Yeah! Not too bad 🎉

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
