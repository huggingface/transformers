---
language: malay
---

# Bahasa ELECTRA Model

Pretrained ELECTRA small language model for Malay and Indonesian. 

## Pretraining Corpus

`electra-small-generator-bahasa-cased` model was pretrained on ~1.8 Billion words. We trained on both standard and social media language structures, and below is list of data we trained on,

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

- This model was trained using Google ELECTRA's github [repository](https://github.com/google-research/electra) on a single TESLA V100 32GB VRAM.
- All steps can reproduce from here, [Malaya/pretrained-model/electra](https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/electra).

## Load Pretrained Model

You can use this model by installing `torch` or `tensorflow` and Huggingface library `transformers`. And you can use it directly by initializing it like this:  

```python
from transformers import ElectraTokenizer, ElectraModel

model = ElectraModel.from_pretrained('huseinzol05/electra-small-generator-bahasa-cased')
tokenizer = ElectraTokenizer.from_pretrained(
    'huseinzol05/electra-small-generator-bahasa-cased',
    do_lower_case = False,
)
```

## Example using AutoModelWithLMHead

```python
from transformers import ElectraTokenizer, AutoModelWithLMHead, pipeline

model = AutoModelWithLMHead.from_pretrained('huseinzol05/electra-small-generator-bahasa-cased')
tokenizer = ElectraTokenizer.from_pretrained(
    'huseinzol05/electra-small-generator-bahasa-cased',
    do_lower_case = False,
)
fill_mask = pipeline('fill-mask', model = model, tokenizer = tokenizer)
print(fill_mask('makan ayam dengan [MASK]'))
```

Output is,

```text
[{'sequence': '[CLS] makan ayam dengan ayam [SEP]',
  'score': 0.08424834907054901,
  'token': 3255},
 {'sequence': '[CLS] makan ayam dengan rendang [SEP]',
  'score': 0.064150370657444,
  'token': 6288},
 {'sequence': '[CLS] makan ayam dengan nasi [SEP]',
  'score': 0.033446669578552246,
  'token': 2533},
 {'sequence': '[CLS] makan ayam dengan kucing [SEP]',
  'score': 0.02803465723991394,
  'token': 3577},
 {'sequence': '[CLS] makan ayam dengan telur [SEP]',
  'score': 0.026627106592059135,
  'token': 6350}]
```

## Results

For further details on the model performance, simply checkout accuracy page from Malaya, https://malaya.readthedocs.io/en/latest/Accuracy.html, we compared with traditional models.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train ELECTRA for Bahasa. 


