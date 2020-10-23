---
language: it
datasets:
- xtreme
---

# Italian-Bert  (Italian Bert) + POS 🎃🏷

This model is a fine-tuned on [xtreme udpos Italian](https://huggingface.co/nlp/viewer/?dataset=xtreme&config=udpos.Italian) version of [Bert Base Italian](https://huggingface.co/dbmdz/bert-base-italian-cased) for **POS** downstream task.

## Details of the downstream task (POS) - Dataset

- [Dataset: xtreme udpos Italian](https://huggingface.co/nlp/viewer/?dataset=xtreme&config=udpos.Italian) 📚

| Dataset                | # Examples |
| ---------------------- | ----- |
| Train                  | 716 K |
| Dev                    | 85 K |

- [Fine-tune on NER script provided by @stefan-it](https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py)

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
X
```

## Metrics on evaluation set 🧾

|                                                      Metric                                                       |  # score  |
| :------------------------------------------------------------------------------------: | :-------: |
| F1                                       | **97.25**  
| Precision                                | **97.15** | 
| Recall                                   | **97.36** |    

## Model in action 🔨


Example of usage

```python
from transformers import pipeline

nlp_pos = pipeline(
    "ner",
    model="sachaarbonel/bert-italian-cased-finetuned-pos",
    tokenizer=(
        'sachaarbonel/bert-spanish-cased-finetuned-pos',  
        {"use_fast": False}
))


text = 'Roma è la Capitale d'Italia.'

nlp_pos(text)
      
'''
Output:
--------
[{'entity': 'PROPN', 'index': 1, 'score': 0.9995346665382385, 'word': 'roma'},
 {'entity': 'AUX', 'index': 2, 'score': 0.9966597557067871, 'word': 'e'},
 {'entity': 'DET', 'index': 3, 'score': 0.9994786977767944, 'word': 'la'},
 {'entity': 'NOUN',
  'index': 4,
  'score': 0.9995198249816895,
  'word': 'capitale'},
 {'entity': 'ADP', 'index': 5, 'score': 0.9990894198417664, 'word': 'd'},
 {'entity': 'PART', 'index': 6, 'score': 0.57159024477005, 'word': "'"},
 {'entity': 'PROPN',
  'index': 7,
  'score': 0.9994804263114929,
  'word': 'italia'},
 {'entity': 'PUNCT', 'index': 8, 'score': 0.9772886633872986, 'word': '.'}]
'''
```
Yeah! Not too bad 🎉

> Created by [Sacha Arbonel/@sachaarbonel](https://twitter.com/sachaarbonel) | [LinkedIn](https://www.linkedin.com/in/sacha-arbonel)

> Made with <span style="color: #e25555;">&hearts;</span> in Paris
