---
language: de
---

# German BERT + LER (Legal Entity Recognition) ⚖️

German BERT ([BERT-base-german-cased](https://huggingface.co/bert-base-german-cased)) fine-tuned on [Legal-Entity-Recognition](https://github.com/elenanereiss/Legal-Entity-Recognition) dataset for **LER** (NER) downstream task.

## Details of the downstream task (NER) - Dataset

[Legal-Entity-Recognition](https://github.com/elenanereiss/Legal-Entity-Recognition): Fine-grained Named Entity Recognition in Legal Documents.

Court decisions from 2017 and 2018 were selected for the dataset, published online by the [Federal Ministry of Justice and Consumer Protection](http://www.rechtsprechung-im-internet.de). The documents originate from seven federal courts: Federal Labour Court (BAG), Federal Fiscal Court (BFH), Federal Court of Justice (BGH), Federal Patent Court (BPatG), Federal Social Court (BSG), Federal Constitutional Court (BVerfG) and Federal Administrative Court (BVerwG). 


|  Split             | # Samples |
| ---------------------- | ----- |
| Train                  | 1657048 |
| Eval                    | 500000 |

- Training script: [Fine-tuning script for NER provided by Huggingface](https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py)
Colab: [How to fine-tune a model for NER using HF scripts](https://colab.research.google.com/drive/156Qrd7NsUHwA3nmQ6gXdZY0NzOvqk9AT?usp=sharing)

- Labels covered (and its distribution):

```
    107 B-AN
    918 B-EUN
   2238 B-GRT
  13282 B-GS
   1113 B-INN
    704 B-LD
    151 B-LDS
   2490 B-LIT
    282 B-MRK
    890 B-ORG
   1374 B-PER
   1480 B-RR
  10046 B-RS
    401 B-ST
     68 B-STR
   1011 B-UN
    282 B-VO
    391 B-VS
   2648 B-VT
     46 I-AN
   6925 I-EUN
   1957 I-GRT
  70257 I-GS
   2931 I-INN
    153 I-LD
     26 I-LDS
  28881 I-LIT
    383 I-MRK
   1185 I-ORG
    330 I-PER
    106 I-RR
 138938 I-RS
     34 I-ST
     55 I-STR
   1259 I-UN
   1572 I-VO
   2488 I-VS
  11121 I-VT
1348525 O
```
- [Annotation Guidelines (German)](https://github.com/elenanereiss/Legal-Entity-Recognition/blob/master/docs/Annotationsrichtlinien.pdf)


## Metrics on evaluation set

|                                                      Metric                                                       |  # score  |
| :------------------------------------------------------------------------------------: | :-------: |
| F1                                       | **85.67**  
| Precision                                | **84.35** | 
| Recall                                   | **87.04** | 
| Accuracy                                 | **98.46** |

## Model in action

Fast usage with **pipelines**:

```python
from transformers import pipeline

nlp_ler = pipeline(
    "ner",
    model="mrm8488/bert-base-german-finetuned-ler",
    tokenizer="mrm8488/bert-base-german-finetuned-ler"
)

text = "Your German legal text here"

nlp_ler(text)

```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
