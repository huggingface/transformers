---
language: malay
---

# Bahasa Tiny-BERT Model

General Distilled Tiny BERT language model for Malay and Indonesian. 

## Pretraining Corpus

`tiny-bert-bahasa-cased` model was distilled on ~1.8 Billion words. We distilled on both standard and social media language structures, and below is list of data we distilled on,

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

## Distilling details

- This model was distilled using huawei-noah Tiny-BERT's github [repository](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) on 3 Titan V100 32GB VRAM.
- All steps can reproduce from here, [Malaya/pretrained-model/tiny-bert](https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/tiny-bert).

## Load Distilled Model

You can use this model by installing `torch` or `tensorflow` and Huggingface library `transformers`. And you can use it directly by initializing it like this:  

```python
from transformers import AlbertTokenizer, BertModel

model = BertModel.from_pretrained('huseinzol05/tiny-bert-bahasa-cased')
tokenizer = AlbertTokenizer.from_pretrained(
    'huseinzol05/tiny-bert-bahasa-cased',
    unk_token = '[UNK]',
    pad_token = '[PAD]',
    do_lower_case = False,
)
```

We use [google/sentencepiece](https://github.com/google/sentencepiece) to train the tokenizer, so to use it, need to load from `AlbertTokenizer`.

## Example using AutoModelWithLMHead

```python
from transformers import AlbertTokenizer, AutoModelWithLMHead, pipeline

model = AutoModelWithLMHead.from_pretrained('huseinzol05/tiny-bert-bahasa-cased')
tokenizer = AlbertTokenizer.from_pretrained(
    'huseinzol05/tiny-bert-bahasa-cased',
    unk_token = '[UNK]',
    pad_token = '[PAD]',
    do_lower_case = False,
)
fill_mask = pipeline('fill-mask', model = model, tokenizer = tokenizer)
print(fill_mask('makan ayam dengan [MASK]'))
```

Output is,

```text
[{'sequence': '[CLS] makan ayam dengan berbual[SEP]',
  'score': 0.00015769545279908925,
  'token': 17859},
 {'sequence': '[CLS] makan ayam dengan kembar[SEP]',
  'score': 0.0001448775001335889,
  'token': 8289},
 {'sequence': '[CLS] makan ayam dengan memaklumkan[SEP]',
  'score': 0.00013484008377417922,
  'token': 6881},
 {'sequence': '[CLS] makan ayam dengan Senarai[SEP]',
  'score': 0.00013061291247140616,
  'token': 11698},
 {'sequence': '[CLS] makan ayam dengan Tiga[SEP]',
  'score': 0.00012453157978598028,
  'token': 4232}]
```

## Results

For further details on the model performance, simply checkout accuracy page from Malaya, https://malaya.readthedocs.io/en/latest/Accuracy.html, we compared with traditional models.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train BERT for Bahasa. 


