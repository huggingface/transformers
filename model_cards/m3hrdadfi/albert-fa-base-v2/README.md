---
language: fa
tags:
- albert-persian
- persian-lm
license: apache-2.0
datasets:
- Persian Wikidumps
- MirasText
- BigBang Page
- Chetor
- Eligasht
- DigiMag
- Ted Talks
- Books (Novels, ...)
---

# ALBERT-Persian

## ALBERT-Persian: A Lite BERT for Self-supervised Learning of Language Representations for the Persian Language

## Introduction

ALBERT-Persian trained on a massive amount of public corpora ([Persian Wikidumps](https://dumps.wikimedia.org/fawiki/), [MirasText](https://github.com/miras-tech/MirasText)) and six other manually crawled text data from a various type of websites ([BigBang Page](https://bigbangpage.com/) `scientific`, [Chetor](https://www.chetor.com/) `lifestyle`, [Eligasht](https://www.eligasht.com/Blog/) `itinerary`,  [Digikala](https://www.digikala.com/mag/) `digital magazine`, [Ted Talks](https://www.ted.com/talks) `general conversational`, Books `novels, storybooks, short stories from old to the contemporary era`).



## Intended uses & limitations

You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to
be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?search=albert-fa) to look for
fine-tuned versions on a task that interests you.


### How to use

#### TensorFlow 2.0

```python
from transformers import AutoConfig, AutoTokenizer, TFAutoModel

config = AutoConfig.from_pretrained("m3hrdadfi/albert-fa-base-v2")
tokenizer = AutoTokenizer.from_pretrained("m3hrdadfi/albert-fa-base-v2")
model = TFAutoModel.from_pretrained("m3hrdadfi/albert-fa-base-v2")

text = "ما در هوشواره معتقدیم با انتقال صحیح دانش و آگاهی، همه افراد می‌توانند از ابزارهای هوشمند استفاده کنند. شعار ما هوش مصنوعی برای همه است."
tokenizer.tokenize(text)

>>> ['▁ما', '▁در', '▁هوش', 'واره', '▁معتقد', 'یم', '▁با', '▁انتقال', '▁صحیح', '▁دانش', '▁و', '▁اگاه', 'ی', '،', '▁همه', '▁افراد', '▁می', '▁توانند', '▁از', '▁ابزارهای', '▁هوشمند', '▁استفاده', '▁کنند', '.', '▁شعار', '▁ما', '▁هوش', '▁مصنوعی', '▁برای', '▁همه', '▁است', '.']

```

#### Pytorch

```python
from transformers import AutoConfig, AutoTokenizer, AutoModel

config = AutoConfig.from_pretrained("m3hrdadfi/albert-fa-base-v2")
tokenizer = AutoTokenizer.from_pretrained("m3hrdadfi/albert-fa-base-v2")
model = AutoModel.from_pretrained("m3hrdadfi/albert-fa-base-v2")
```

## Training

ALBERT-Persian is the first attempt on ALBERT for the Persian Language. The model was trained based on Google's ALBERT BASE Version 2.0 over various writing styles from numerous subjects (e.g., scientific, novels, news) with more than `3.9M` documents, `73M` sentences, and `1.3B` words, like the way we did for [ParsBERT](https://github.com/hooshvare/parsbert).

## Goals
Objective goals during training are as below (after 140K steps).

``` bash
***** Eval results *****
global_step = 140000
loss = 2.0080082
masked_lm_accuracy = 0.6141017
masked_lm_loss = 1.9963315
sentence_order_accuracy = 0.985
sentence_order_loss = 0.06908702
```


## Derivative models

### Base Config

#### Albert Model
- [m3hrdadfi/albert-face-base-v2](https://huggingface.co/m3hrdadfi/albert-fa-base-v2) 

#### Albert Sentiment Analysis
- [m3hrdadfi/albert-fa-base-v2-sentiment-digikala](https://huggingface.co/m3hrdadfi/albert-fa-base-v2-sentiment-digikala) 
- [m3hrdadfi/albert-fa-base-v2-sentiment-snappfood](https://huggingface.co/m3hrdadfi/albert-fa-base-v2-sentiment-snappfood) 
- [m3hrdadfi/albert-fa-base-v2-sentiment-deepsentipers-binary](https://huggingface.co/m3hrdadfi/albert-fa-base-v2-sentiment-deepsentipers-binary) 
- [m3hrdadfi/albert-fa-base-v2-sentiment-deepsentipers-multi](https://huggingface.co/m3hrdadfi/albert-fa-base-v2-sentiment-deepsentipers-multi) 
- [m3hrdadfi/albert-fa-base-v2-sentiment-binary](https://huggingface.co/m3hrdadfi/albert-fa-base-v2-sentiment-binary) 
- [m3hrdadfi/albert-fa-base-v2-sentiment-multi](https://huggingface.co/m3hrdadfi/albert-fa-base-v2-sentiment-multi) 
- [m3hrdadfi/albert-fa-base-v2-sentiment-multi](https://huggingface.co/m3hrdadfi/albert-fa-base-v2-sentiment-multi) 

#### Albert Text Classification
- [m3hrdadfi/albert-fa-base-v2-clf-digimag](https://huggingface.co/m3hrdadfi/albert-fa-base-v2-clf-digimag) 
- [m3hrdadfi/albert-fa-base-v2-clf-persiannews](https://huggingface.co/m3hrdadfi/albert-fa-base-v2-clf-persiannews) 

#### Albert NER
- [m3hrdadfi/albert-fa-base-v2-ner](https://huggingface.co/m3hrdadfi/albert-fa-base-v2-ner) 
- [m3hrdadfi/albert-fa-base-v2-ner-arman](https://huggingface.co/m3hrdadfi/albert-fa-base-v2-ner-arman) 
- [m3hrdadfi/albert-fa-base-v2-ner-arman](https://huggingface.co/m3hrdadfi/albert-fa-base-v2-ner-arman) 

## Eval results

The following tables summarize the F1 scores obtained by ALBERT-Persian as compared to other models and architectures.


### Sentiment Analysis (SA) Task

|          Dataset         | ALBERT-fa-base-v2 | ParsBERT-v1 | mBERT | DeepSentiPers |
|:------------------------:|:-----------------:|:-----------:|:-----:|:-------------:|
|  Digikala User Comments  |       81.12       |    81.74    | 80.74 |       -       |
|  SnappFood User Comments |       85.79       |    88.12    | 87.87 |       -       |
|  SentiPers (Multi Class) |       66.12       |    71.11    |   -   |     69.33     |
| SentiPers (Binary Class) |       91.09       |    92.13    |   -   |     91.98     |


### Text Classification (TC) Task

|      Dataset      | ALBERT-fa-base-v2 | ParsBERT-v1 | mBERT |
|:-----------------:|:-----------------:|:-----------:|:-----:|
| Digikala Magazine |       92.33       |    93.59    | 90.72 |
|    Persian News   |       97.01       |    97.19    | 95.79 |


### Named Entity Recognition (NER) Task

| Dataset | ALBERT-fa-base-v2 | ParsBERT-v1 | mBERT | MorphoBERT | Beheshti-NER | LSTM-CRF | Rule-Based CRF | BiLSTM-CRF |
|:-------:|:-----------------:|:-----------:|:-----:|:----------:|:------------:|:--------:|:--------------:|:----------:|
|  PEYMA  |       88.99       |    93.10    | 86.64 |      -     |     90.59    |     -    |      84.00     |      -     |
|  ARMAN  |       97.43       |    98.79    | 95.89 |    89.9    |     84.03    |   86.55  |        -       |    77.45   |


### BibTeX entry and citation info

Please cite in publications as the following:

```bibtex
@misc{ALBERT-Persian,
  author = {Mehrdad Farahani},
  title = {ALBERT-Persian: A Lite BERT for Self-supervised Learning of Language Representations for the Persian Language},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/m3hrdadfi/albert-persian}},
}

@article{ParsBERT,
    title={ParsBERT: Transformer-based Model for Persian Language Understanding},
    author={Mehrdad Farahani, Mohammad Gharachorloo, Marzieh Farahani, Mohammad Manthouri},
    journal={ArXiv},
    year={2020},
    volume={abs/2005.12515}
}
```

## Questions?
Post a Github issue on the [ALBERT-Persian](https://github.com/m3hrdadfi/albert-persian) repo.
