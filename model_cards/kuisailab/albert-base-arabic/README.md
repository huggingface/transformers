---
language: ar
datasets:
- oscar
- wikipedia
tags:
- ar
- masked-lm
- lm-head
---


# Arabic-ALBERT Base

Arabic edition of ALBERT Base pretrained language model

## Pretraining data

The models were pretrained on ~4.4 Billion words:

- Arabic version of [OSCAR](https://oscar-corpus.com/) (unshuffled version of the corpus) - filtered from [Common Crawl](http://commoncrawl.org/)
- Recent dump of Arabic [Wikipedia](https://dumps.wikimedia.org/backup-index.html)

__Notes on training data:__

- Our final version of corpus contains some non-Arabic words inlines, which we did not remove from sentences since that would affect some tasks like NER.
- Although non-Arabic characters were lowered as a preprocessing step, since Arabic characters do not have upper or lower case, there is no cased and uncased version of the model.
- The corpus and vocabulary set are not restricted to Modern Standard Arabic, they contain some dialectical Arabic too.

## Pretraining details

- These models were trained using Google ALBERT's github [repository](https://github.com/google-research/albert) on a single TPU v3-8 provided for free from [TFRC](https://www.tensorflow.org/tfrc).
- Our pretraining procedure follows training settings of bert with some changes: trained for 7M training steps with batchsize of 64, instead of 125K with batchsize of 4096.

## Models

|  | albert-base | albert-large | albert-xlarge |
|:---:|:---:|:---:|:---:|
| Hidden Layers | 12 | 24 | 24 |
| Attention heads | 12 | 16 | 32 |
| Hidden size | 768 | 1024 | 2048 |

## Results

For further details on the models performance or any other queries, please refer to [Arabic-ALBERT](https://github.com/KUIS-AI-Lab/Arabic-ALBERT/)

## How to use

You can use these models by installing `torch` or `tensorflow` and Huggingface library `transformers`. And you can use it directly by initializing it like this:  

```python

from transformers import AutoTokenizer, AutoModel

# loading the tokenizer
base_tokenizer    = AutoTokenizer.from_pretrained("kuisailab/albert-base-arabic")

# loading the model
base_model   = AutoModel.from_pretrained("kuisailab/albert-base-arabic")

```

## Acknowledgement

Thanks to Google for providing free TPU for the training process and for Huggingface for hosting these models on their servers 😊
