---
language: spanish
thumbnail:
---

# Spanish BERT (BETO) + Syntax POS tagging ✍🏷

This model is a fine-tuned version of the Spanish BERT [(BETO)](https://github.com/dccuchile/beto) on Spanish **syntax** annotations in [CONLL CORPORA](https://www.kaggle.com/nltkdata/conll-corpora) dataset for **syntax POS** (Part of Speech tagging) downstream task.

## Details of the downstream task (Syntax POS) - Dataset

- [Dataset: CONLL Corpora ES](https://www.kaggle.com/nltkdata/conll-corpora)

#### [Fine-tune script on NER dataset provided by Huggingface](https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py)

#### 21 Syntax annotations (Labels) covered:

- \_
- ATR
- ATR.d
- CAG
- CC
- CD
- CD.Q
- CI
- CPRED
- CPRED.CD
- CPRED.SUJ
- CREG
- ET
- IMPERS
- MOD
- NEG
- PASS
- PUNC
- ROOT
- SUJ
- VOC

## Metrics on test set 📋

|  Metric   |  # score  |
| :-------: | :-------: |
|    F1     | **89.27** |
| Precision | **89.44** |
|  Recall   | **89.11** |

## Model in action 🔨

Fast usage with **pipelines** 🧪

```python
from transformers import pipeline

nlp_pos_syntax = pipeline(
    "ner",
    model="mrm8488/bert-spanish-cased-finetuned-pos-syntax",
    tokenizer="mrm8488/bert-spanish-cased-finetuned-pos-syntax"
)

text = 'Mis amigos están pensando viajar a Londres este verano.'

nlp_pos_syntax(text)[1:len(nlp_pos_syntax(text))-1]
```

```json
[
  { "entity": "_", "score": 0.9999216794967651, "word": "Mis" },
  { "entity": "SUJ", "score": 0.999882698059082, "word": "amigos" },
  { "entity": "_", "score": 0.9998869299888611, "word": "están" },
  { "entity": "ROOT", "score": 0.9980518221855164, "word": "pensando" },
  { "entity": "_", "score": 0.9998420476913452, "word": "viajar" },
  { "entity": "CD", "score": 0.999351978302002, "word": "a" },
  { "entity": "_", "score": 0.999959409236908, "word": "Londres" },
  { "entity": "_", "score": 0.9998968839645386, "word": "este" },
  { "entity": "CC", "score": 0.99931401014328, "word": "verano" },
  { "entity": "PUNC", "score": 0.9998534917831421, "word": "." }
]
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
