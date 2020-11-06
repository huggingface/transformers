---
language: de
---

# German Medical BERT

This is a fine-tuned model on Medical domain for German language and based on German BERT.

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
![performance](https://raw.githubusercontent.com/smanjil/finetune-lm/master/performance.png)

## Author
Manjil Shrestha: `shresthamanjil21 [at] gmail.com`

Get in touch:
[LinkedIn](https://www.linkedin.com/in/manjil-shrestha-038527b4/)
