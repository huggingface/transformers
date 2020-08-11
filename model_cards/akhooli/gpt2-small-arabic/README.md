---
language: "ar"
datasets:
- Arabic Wikipedia
metrics:
- none
---

# GPT2-Small-Arabic

## Model description

GPT2 model from Arabic Wikipedia dataset based on gpt2-small (using Fastai2).

## Intended uses & limitations

#### How to use

An example is provided in this [colab notebook](https://colab.research.google.com/drive/1mRl7c-5v-Klx27EEAEOAbrfkustL4g7a?usp=sharing). 
Both text and poetry (fine-tuned model) generation are included.

#### Limitations and bias

GPT2-small-arabic (trained on Arabic Wikipedia) has several limitations in terms of coverage (Arabic Wikipeedia quality, no diacritics) and training performance. 
Use as demonstration or proof of concepts but not as production code.

## Training data

This pretrained model used the Arabic Wikipedia dump (around 900 MB). 

## Training procedure

Training was done using [Fastai2](https://github.com/fastai/fastai2/) library on Kaggle, using free GPU.

## Eval results 
Final perplexity reached was 72.19,  loss: 4.28, accuracy: 0.307

### BibTeX entry and citation info

```bibtex
@inproceedings{Abed Khooli,
  year={2020}
}
```
