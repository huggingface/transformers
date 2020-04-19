---
language: malay
---

# Bahasa ELECTRA Model

Pretrained ELECTRA base language model for Malay and Indonesian. 

## Pretraining Corpus

`electra-base-discriminator-bahasa-cased` model was pretrained on ~1.8 Billion words. We trained on both standard and social media language structures, and below is list of data we trained on,

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

model = ElectraModel.from_pretrained('huseinzol05/electra-base-discriminator-bahasa-cased')
tokenizer = ElectraTokenizer.from_pretrained(
    'huseinzol05/electra-base-discriminator-bahasa-cased',
    do_lower_case = False,
)
```

## Example using ElectraForPreTraining

```python
from transformers import ElectraTokenizer, AutoModelWithLMHead, pipeline

model = ElectraForPreTraining.from_pretrained('huseinzol05/electra-base-discriminator-bahasa-cased')
tokenizer = ElectraTokenizer.from_pretrained(
    'huseinzol05/electra-base-discriminator-bahasa-cased', 
    do_lower_case = False
)
sentence = 'kerajaan sangat prihatin terhadap rakyat'
fake_tokens = tokenizer.tokenize(sentence)
fake_inputs = tokenizer.encode(sentence, return_tensors="pt")
discriminator_outputs = discriminator(fake_inputs)
predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

list(zip(fake_tokens, predictions.tolist()))
```

Output is,

```text
[('kerajaan', 0.0),
 ('sangat', 0.0),
 ('prihatin', 0.0),
 ('terhadap', 0.0),
 ('rakyat', 0.0)]
```

## Results

For further details on the model performance, simply checkout accuracy page from Malaya, https://malaya.readthedocs.io/en/latest/Accuracy.html, we compared with traditional models.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train ELECTRA for Bahasa. 


