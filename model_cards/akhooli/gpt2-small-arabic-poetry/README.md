---
language: "ar"
tags:
- text-generation
license: ""
datasets:
- Arabic poetry from several eras
---

# GPT2-Small-Arabic-Poetry

## Model description

Fine-tuned model of Arabic poetry dataset based on gpt2-small-arabic.

## Intended uses & limitations

#### How to use

An example is provided in this [colab notebook](https://colab.research.google.com/drive/1mRl7c-5v-Klx27EEAEOAbrfkustL4g7a?usp=sharing).

#### Limitations and bias

Both the GPT2-small-arabic (trained on Arabic Wikipedia) and this model have several limitations in terms of coverage and training performance. 
Use them as demonstrations or proof of concepts but not as production code.

## Training data

This pretrained model used the [Arabic Poetry dataset](https://www.kaggle.com/ahmedabelal/arabic-poetry) from 9 different eras with a total of around 40k poems. 
The dataset was trained (fine-tuned) based on the [gpt2-small-arabic](https://huggingface.co/akhooli/gpt2-small-arabic) transformer model.

## Training procedure

Training was done using [Simple Transformers](https://github.com/ThilinaRajapakse/simpletransformers) library on Kaggle, using free GPU.

## Eval results 
Final perplexity reached ws 76.3, loss: 4.33

### BibTeX entry and citation info

```bibtex
@inproceedings{Abed Khooli,
  year={2020}
}
```
