---
language: id
datasets:
- oscar
---
# IndoBERT (Indonesian BERT Model)

## Model description
IndoBERT is a pre-trained language model based on BERT architecture for the Indonesian Language. 

This model is base-uncased version which use bert-base config.

## Intended uses & limitations

#### How to use

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("sarahlintang/IndoBERT")
model = AutoModel.from_pretrained("sarahlintang/IndoBERT")
tokenizer.encode("hai aku mau makan.")
[2, 8078, 1785, 2318, 1946, 18, 4]
```


## Training data

This model was pre-trained on 16 GB of raw text ~2 B words from Oscar Corpus (https://oscar-corpus.com/). 

This model is equal to bert-base model which has 32,000 vocabulary size. 

## Training procedure

The training of the model has been performed using Google’s original Tensorflow code on eight core Google Cloud TPU v2.
We used a Google Cloud Storage bucket, for persistent storage of training data and models.

## Eval results

We evaluate this model on three Indonesian NLP downstream task:
- some extractive summarization model
- sentiment analysis
- Part-of-Speech Tagger
it was proven that this model outperforms multilingual BERT for all downstream tasks.
