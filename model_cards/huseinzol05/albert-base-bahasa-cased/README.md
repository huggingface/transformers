---
language: malay
---

# Bahasa Albert Model

Pretrained Albert base language model for Malay and Indonesian. 

## Pretraining Corpus

`albert-base-bahasa-cased` model was pretrained on ~1.8 Billion words. We trained on both standard and social media language structures, and below is list of data we trained on,

1. [dumping wikipedia](https://github.com/huseinzol05/Malaya-Dataset#wikipedia-1).
2. [local instagram](https://github.com/huseinzol05/Malaya-Dataset#instagram).
3. [local twitter](https://github.com/huseinzol05/Malaya-Dataset#twitter-1).
4. [local news](https://github.com/huseinzol05/Malaya-Dataset#public-news).
5. [local parliament text](https://github.com/huseinzol05/Malaya-Dataset#parliament).
6. [local singlish/manglish text](https://github.com/huseinzol05/Malaya-Dataset#singlish-text).
7. [IIUM Confession](https://github.com/huseinzol05/Malaya-Dataset#iium-confession).
8. [Wattpad](https://github.com/huseinzol05/Malaya-Dataset#wattpad).
9. [Academia PDF](https://github.com/huseinzol05/Malaya-Dataset#academia-pdf).

Preprocessing steps can reproduce from here, [Malaya/pretrained-model/preprocess](https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/preprocess).

## Pretraining details

- This model was trained using Google Albert's github [repository](https://github.com/google-research/ALBERT) on v3-8 TPU.
- All steps can reproduce from here, [Malaya/pretrained-model/albert](https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/albert).

## Load Pretrained Model

You can use this model by installing `torch` or `tensorflow` and Huggingface library `transformers`. And you can use it directly by initializing it like this:  

```python
from transformers import AlbertTokenizer, AlbertModel

model = BertModel.from_pretrained('huseinzol05/albert-base-bahasa-cased')
tokenizer = AlbertTokenizer.from_pretrained(
    'huseinzol05/albert-base-bahasa-cased',
    do_lower_case = False,
)
```

## Example using AutoModelWithLMHead

```python
from transformers import AlbertTokenizer, AutoModelWithLMHead, pipeline

model = AutoModelWithLMHead.from_pretrained('huseinzol05/albert-base-bahasa-cased')
tokenizer = AlbertTokenizer.from_pretrained(
    'huseinzol05/albert-base-bahasa-cased',
    do_lower_case = False,
)
fill_mask = pipeline('fill-mask', model = model, tokenizer = tokenizer)
print(fill_mask('makan ayam dengan [MASK]'))
```

Output is,

```text
[{'sequence': '[CLS] makan ayam dengan ayam[SEP]',
  'score': 0.044952988624572754,
  'token': 629},
 {'sequence': '[CLS] makan ayam dengan sayur[SEP]',
  'score': 0.03621877357363701,
  'token': 1639},
 {'sequence': '[CLS] makan ayam dengan ikan[SEP]',
  'score': 0.034429922699928284,
  'token': 758},
 {'sequence': '[CLS] makan ayam dengan nasi[SEP]',
  'score': 0.032447945326566696,
  'token': 453},
 {'sequence': '[CLS] makan ayam dengan rendang[SEP]',
  'score': 0.028885239735245705,
  'token': 2451}]
```

## Results

For further details on the model performance, simply checkout accuracy page from Malaya, https://malaya.readthedocs.io/en/latest/Accuracy.html, we compared with traditional models.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train Albert for Bahasa. 


