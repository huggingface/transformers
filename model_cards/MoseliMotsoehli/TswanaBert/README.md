---
language: setswana
---

# TswanaBert

## Model Description.
TswanaBERT is a transformers model pretrained on a corpus of Setswana data in a self-supervised fashion by masking part of the input words and training to predict the masks.

## Intended uses & limitations
The model can  be used for either masked language modeling or next word prediction. it can also be fine-tuned for a specifict application. 

#### How to use

```python
>>> from transformers import pipeline
>>> from transformers import AutoTokenizer, AutoModelWithLMHead

>>> tokenizer = AutoTokenizer.from_pretrained("MoseliMotsoehli/TswanaBert")
>>> model = AutoModelWithLMHead.from_pretrained("MoseliMotsoehli/TswanaBert")
>>> unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
>>> unmasker("Ntshopotse <mask> e godile.")

[{'score': 0.32749542593955994,
  'sequence': '<s>Ntshopotse setse e godile.</s>',
  'token': 538,
  'token_str': 'Ġsetse'},
 {'score': 0.060260992497205734,
  'sequence': '<s>Ntshopotse le e godile.</s>',
  'token': 270,
  'token_str': 'Ġle'},
 {'score': 0.058460816740989685,
  'sequence': '<s>Ntshopotse bone e godile.</s>',
  'token': 364,
  'token_str': 'Ġbone'},
 {'score': 0.05694682151079178,
  'sequence': '<s>Ntshopotse ga e godile.</s>',
  'token': 298,
  'token_str': 'Ġga'},
 {'score': 0.0565204992890358,
  'sequence': '<s>Ntshopotse, e godile.</s>',
  'token': 16,
  'token_str': ','}]
```

#### Limitations and bias
The model is trained on a fairly small collection of setwana, mostly from news articles and creative writtings, and so is not representative enough of the language as yet.

## Training data

The largest portion of this dataset (10k)  lines of text, comes from the [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download)

The I then added 200 more phrases and sentences by scrapping following sites. I continue to expand the dataset

* http://setswana.blogspot.com/
* https://omniglot.com/writing/tswana.php
* http://www.dailynews.gov.bw/
* http://www.mmegi.bw/index.php
* https://tsena.co.bw
* http://www.botswana.co.za/Cultural_Issues-travel/botswana-country-guide-en-route.html

## Training procedure
The model was trained on a google colab Tesla T4 GPU for 200 epochs with a batch size of 64, on 13446 learned tokens.
Other model training configuration setting can be found [here](https://s3.amazonaws.com/models.huggingface.co/bert/MoseliMotsoehli/TswanaBert/config.json)

### BibTeX entry and citation info

```bibtex
@inproceedings{author = {Moseli Motsoehli},
  year={2020}
}
```
