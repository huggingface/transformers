---
language: fa
tags:
- bert-fa
- bert-persian
- persian-lm
license: apache-2.0
---

# ParsBERT (v2.0)
A Transformer-based Model for Persian Language Understanding


We reconstructed the vocabulary and fine-tuned the ParsBERT v1.1 on the new Persian corpora in order to provide some functionalities for using ParsBERT in other scopes!
Please follow the [ParsBERT](https://github.com/hooshvare/parsbert) repo for the latest information about previous and current models.

## Introduction

ParsBERT is a monolingual language model based on Google’s BERT architecture. This model is pre-trained on large Persian corpora with various writing styles from numerous subjects (e.g., scientific, novels, news) with more than `3.9M` documents, `73M` sentences, and `1.3B` words.
 
Paper presenting ParsBERT: [arXiv:2005.12515](https://arxiv.org/abs/2005.12515)

## Intended uses & limitations

You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to
be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?search=bert-fa) to look for
fine-tuned versions on a task that interests you.


### How to use

#### TensorFlow 2.0

```python
from transformers import AutoConfig, AutoTokenizer, TFAutoModel

config = AutoConfig.from_pretrained("HooshvareLab/bert-fa-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
model = TFAutoModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")

text = "ما در هوشواره معتقدیم با انتقال صحیح دانش و آگاهی، همه افراد میتوانند از ابزارهای هوشمند استفاده کنند. شعار ما هوش مصنوعی برای همه است."
tokenizer.tokenize(text)

>>> ['ما', 'در', 'هوش', '##واره', 'معتقدیم', 'با', 'انتقال', 'صحیح', 'دانش', 'و', 'اگاهی', '،', 'همه', 'افراد', 'میتوانند', 'از', 'ابزارهای', 'هوشمند', 'استفاده', 'کنند', '.', 'شعار', 'ما', 'هوش', 'مصنوعی', 'برای', 'همه', 'است', '.']
```

#### Pytorch

```python
from transformers import AutoConfig, AutoTokenizer, AutoModel

config = AutoConfig.from_pretrained("HooshvareLab/bert-fa-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
model = AutoModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")
```

## Training

ParsBERT trained on a massive amount of public corpora ([Persian Wikidumps](https://dumps.wikimedia.org/fawiki/), [MirasText](https://github.com/miras-tech/MirasText)) and six other manually crawled text data from a various type of websites ([BigBang Page](https://bigbangpage.com/) `scientific`, [Chetor](https://www.chetor.com/) `lifestyle`, [Eligasht](https://www.eligasht.com/Blog/) `itinerary`,  [Digikala](https://www.digikala.com/mag/) `digital magazine`, [Ted Talks](https://www.ted.com/talks) `general conversational`, Books `novels, storybooks, short stories from old to the contemporary era`).

As a part of ParsBERT methodology, an extensive pre-processing combining POS tagging and WordPiece segmentation was carried out to bring the corpora into a proper format.

## Goals
Objective goals during training are as below (after 300k steps).

``` bash
***** Eval results *****
global_step = 300000
loss = 1.4392426
masked_lm_accuracy = 0.6865794
masked_lm_loss = 1.4469004
next_sentence_accuracy = 1.0
next_sentence_loss = 6.534152e-05
```


## Derivative models

### Base Config

#### ParsBERT v2.0 Model
- [HooshvareLab/bert-fa-base-uncased](https://huggingface.co/HooshvareLab/bert-fa-base-uncased) 

#### ParsBERT v2.0 Sentiment Analysis
- [HooshvareLab/bert-fa-base-uncased-sentiment-digikala](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-digikala) 
- [HooshvareLab/bert-fa-base-uncased-sentiment-snappfood](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-snappfood) 
- [HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-binary](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-binary) 
- [HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-multi](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-multi) 

#### ParsBERT v2.0 Text Classification
- [HooshvareLab/bert-fa-base-uncased-clf-digimag](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-clf-digimag) 
- [HooshvareLab/bert-fa-base-uncased-clf-persiannews](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-clf-persiannews) 

#### ParsBERT v2.0 NER 
- [HooshvareLab/bert-fa-base-uncased-ner-peyma](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-ner-peyma) 
- [HooshvareLab/bert-fa-base-uncased-ner-arman](https://huggingface.co/HooshvareLab/bert-fa-base-uncased-ner-arman) 


## Eval results

ParsBERT is evaluated on three NLP downstream tasks: Sentiment Analysis (SA), Text Classification, and Named Entity Recognition (NER). For this matter and due to insufficient resources, two large datasets for SA and two for text classification were manually composed, which are available for public use and benchmarking. ParsBERT outperformed all other language models, including multilingual BERT and other hybrid deep learning models for all tasks, improving the state-of-the-art performance in Persian language modeling.


### Sentiment Analysis (SA) Task

|          Dataset         | ParsBERT v2 | ParsBERT v1 | mBERT | DeepSentiPers |
|:------------------------:|:-----------:|:-----------:|:-----:|:-------------:|
|  Digikala User Comments  |    81.72    |    81.74*   | 80.74 |       -       |
|  SnappFood User Comments |    87.98    |    88.12*   | 87.87 |       -       |
|  SentiPers (Multi Class) |    71.31*   |    71.11    |   -   |     69.33     |
| SentiPers (Binary Class) |    92.42*   |    92.13    |   -   |     91.98     |


### Text Classification (TC) Task

|      Dataset      | ParsBERT v2 | ParsBERT v1 | mBERT |
|:-----------------:|:-----------:|:-----------:|:-----:|
| Digikala Magazine |    93.65*   |    93.59    | 90.72 |
|    Persian News   |    97.44*   |    97.19    | 95.79 |


### Named Entity Recognition (NER) Task

| Dataset | ParsBERT v2 | ParsBERT v1 | mBERT | MorphoBERT | Beheshti-NER | LSTM-CRF | Rule-Based CRF | BiLSTM-CRF |
|:-------:|:-----------:|:-----------:|:-----:|:----------:|:------------:|:--------:|:--------------:|:----------:|
|  PEYMA  |    93.40*   |    93.10    | 86.64 |      -     |     90.59    |     -    |      84.00     |      -     |
|  ARMAN  |    99.84*   |    98.79    | 95.89 |    89.9    |     84.03    |   86.55  |        -       |    77.45   |




### BibTeX entry and citation info

Please cite in publications as the following:

```bibtex
@article{ParsBERT,
    title={ParsBERT: Transformer-based Model for Persian Language Understanding},
    author={Mehrdad Farahani, Mohammad Gharachorloo, Marzieh Farahani, Mohammad Manthouri},
    journal={ArXiv},
    year={2020},
    volume={abs/2005.12515}
}
```

## Questions?
Post a Github issue on the [ParsBERT Issues](https://github.com/hooshvare/parsbert/issues) repo.
