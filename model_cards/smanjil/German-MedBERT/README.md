---
language: de
tags: 
- exbert
- German
---

<a href="https://huggingface.co/exbert/?model=smanjil/German-MedBERT">
	<img width="300px" src="https://cdn-media.huggingface.co/exbert/button.png">
</a>

# German Medical BERT

This is a fine-tuned model on Medical domain for German language and based on German BERT. This model has only been trained to improve on target task (Masked Language Model). It can later be used to perform a downstream task of your needs, while I performed it for NTS-ICD-10 text classification task.

## Overview
**Language model:** bert-base-german-cased

**Language:** German

**Fine-tuning:** Medical articles (diseases, symptoms, therapies, etc..)

**Eval data:** NTS-ICD-10 dataset (Classification)

**Infrastructure:** Gogle Colab


## Details
- We fine-tuned using Pytorch with Huggingface library on Colab GPU.
- With standard parameter settings for fine-tuning as mentioned in original BERT's paper.
- Although had to train for upto 25 epochs for classification.

## Performance (Micro precision, recall and f1 score for multilabel code classification)

|Models			|P	|R	|F1	|
|:--------------	|:------|:------|:------|
|German BERT		|86.04	|75.82	|80.60	|
|German MedBERT-256	|87.41	|77.97	|82.42	|
|German MedBERT-512	|87.75	|78.26	|82.73	|

## Author
Manjil Shrestha: `shresthamanjil21 [at] gmail.com`

Get in touch:
[LinkedIn](https://www.linkedin.com/in/manjil-shrestha-038527b4/)
