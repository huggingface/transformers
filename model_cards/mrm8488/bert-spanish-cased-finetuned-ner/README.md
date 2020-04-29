---
language: spanish
thumbnail: https://i.imgur.com/jgBdimh.png
---

# Spanish BERT (BETO) + NER

This model is a fine-tuned on [NER-C](https://www.kaggle.com/nltkdata/conll-corpora) version of the Spanish BERT cased [(BETO)](https://github.com/dccuchile/beto) for **NER** downstream task.

## Details of the downstream task (NER) - Dataset

- [Dataset:  CONLL Corpora ES](https://www.kaggle.com/nltkdata/conll-corpora) 

I preprocessed the dataset and splitted it as train / dev (80/20)

| Dataset                | # Examples |
| ---------------------- | ----- |
| Train                  | 8.7 K |
| Dev                    | 2.2 K |


- [Fine-tune on NER script provided by Huggingface](https://github.com/huggingface/transformers/blob/master/examples/run_ner.py)

- Labels covered:

```
B-LOC
B-MISC
B-ORG
B-PER
I-LOC
I-MISC
I-ORG
I-PER
O
```

## Metrics on evaluation set:

|                                                      Metric                                                       |  # score  |
| :------------------------------------------------------------------------------------: | :-------: |
| F1                                       | **90.17**  
| Precision                                | **89.86** | 
| Recall                                   | **90.47** |    

## Comparison:

|                                                      Model                                                       |  # F1 score  |Size(MB)|
| :--------------------------------------------------------------------------------------------------------------: | :-------: |:------|
|                                        bert-base-spanish-wwm-cased (BETO)                                        |   88.43   | 421
| [bert-spanish-cased-finetuned-ner (this one)](https://huggingface.co/mrm8488/bert-spanish-cased-finetuned-ner) | **90.17** | 420 |
|                                              Best Multilingual BERT                                              |   87.38   | 681 |
|[TinyBERT-spanish-uncased-finetuned-ner](https://huggingface.co/mrm8488/TinyBERT-spanish-uncased-finetuned-ner) | 70.00 | **55** |

## Model in action

Fast usage with **pipelines**:

```python
from transformers import pipeline

nlp_ner = pipeline(
    "ner",
    model="mrm8488/bert-spanish-cased-finetuned-ner",
    tokenizer=(
        'mrm8488/bert-spanish-cased-finetuned-ner',  
        {"use_fast": False}
))

text = 'Mis amigos están pensando viajar a Londres este verano'

nlp_ner(text)

#Output: [{'entity': 'B-LOC', 'score': 0.9998720288276672, 'word': 'Londres'}]
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
