---
language: malay
---

# Bahasa BERT Model

Pretrained BERT base language model for Malay and Indonesian. 

## Pretraining Corpus

`bert-base-bahasa-cased` model was pretrained on ~1.8 Billion words. We trained on both standard and social media language structures, and below is list of data we trained on,

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

- This model was trained using Google BERT's github [repository](https://github.com/google-research/bert) on 3 Titan V100 32GB VRAM.
- All steps can reproduce from here, [Malaya/pretrained-model/bert](https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/bert).

## Load Pretrained Model

You can use this model by installing `torch` or `tensorflow` and Huggingface library `transformers`. And you can use it directly by initializing it like this:  

```python
from transformers import AlbertTokenizer, BertModel

model = BertModel.from_pretrained('huseinzol05/bert-base-bahasa-cased')
tokenizer = AlbertTokenizer.from_pretrained(
    'huseinzol05/bert-base-bahasa-cased',
    unk_token = '[UNK]',
    pad_token = '[PAD]',
    do_lower_case = False,
)
```

We use [google/sentencepiece](https://github.com/google/sentencepiece) to train the tokenizer, so to use it, need to load from `AlbertTokenizer`.

## Example using AutoModelWithLMHead

```python
from transformers import AlbertTokenizer, AutoModelWithLMHead, pipeline

model = AutoModelWithLMHead.from_pretrained('huseinzol05/bert-base-bahasa-cased')
tokenizer = AlbertTokenizer.from_pretrained(
    'huseinzol05/bert-base-bahasa-cased',
    unk_token = '[UNK]',
    pad_token = '[PAD]',
    do_lower_case = False,
)
fill_mask = pipeline('fill-mask', model = model, tokenizer = tokenizer)
print(fill_mask('makan ayam dengan [MASK]'))
```

Output is,

```text
[{'sequence': '[CLS] makan ayam dengan rendang[SEP]',
  'score': 0.10812027007341385,
  'token': 2446},
 {'sequence': '[CLS] makan ayam dengan kicap[SEP]',
  'score': 0.07653367519378662,
  'token': 12928},
 {'sequence': '[CLS] makan ayam dengan nasi[SEP]',
  'score': 0.06839974224567413,
  'token': 450},
 {'sequence': '[CLS] makan ayam dengan ayam[SEP]',
  'score': 0.059544261544942856,
  'token': 638},
 {'sequence': '[CLS] makan ayam dengan sayur[SEP]',
  'score': 0.05294966697692871,
  'token': 1639}]
```

## Results

For further details on the model performance, simply checkout accuracy page from Malaya, https://malaya.readthedocs.io/en/latest/Accuracy.html, we compared with traditional models.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train BERT for Bahasa. 


