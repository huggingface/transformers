---
language: en
tags:
- singapore
- sg
- singlish
- malaysia
- ms
- manglish
- albert-base-v2
license: mit
datasets:
- reddit singapore, malaysia
- hardwarezone
widget:
- text: "dont play [MASK] leh"
- text: "die [MASK] must try"
---

# Model name

SingBert Lite - Bert for Singlish (SG) and Manglish (MY).

## Model description

Similar to [SingBert](https://huggingface.co/zanelim/singbert) but the lite-version, which was initialized from [Albert base v2](https://github.com/google-research/albert#albert), with pre-training finetuned on
[singlish](https://en.wikipedia.org/wiki/Singlish) and [manglish](https://en.wikipedia.org/wiki/Manglish) data.

## Intended uses & limitations

#### How to use

```python
>>> from transformers import pipeline
>>> nlp = pipeline('fill-mask', model='zanelim/singbert-lite-sg')
>>> nlp("die [MASK] must try")

[{'sequence': '[CLS] die die must try[SEP]',
  'score': 0.7731555700302124,
  'token': 1327,
  'token_str': '▁die'},
 {'sequence': '[CLS] die also must try[SEP]',
  'score': 0.04763784259557724,
  'token': 67,
  'token_str': '▁also'},
 {'sequence': '[CLS] die still must try[SEP]',
  'score': 0.01859409362077713,
  'token': 174,
  'token_str': '▁still'},
 {'sequence': '[CLS] die u must try[SEP]',
  'score': 0.015824034810066223,
  'token': 287,
  'token_str': '▁u'},
 {'sequence': '[CLS] die is must try[SEP]',
  'score': 0.011271446943283081,
  'token': 25,
  'token_str': '▁is'}]

>>> nlp("dont play [MASK] leh")

[{'sequence': '[CLS] dont play play leh[SEP]',
  'score': 0.4365769624710083,
  'token': 418,
  'token_str': '▁play'},
 {'sequence': '[CLS] dont play punk leh[SEP]',
  'score': 0.06880936771631241,
  'token': 6769,
  'token_str': '▁punk'},
 {'sequence': '[CLS] dont play game leh[SEP]',
  'score': 0.051739856600761414,
  'token': 250,
  'token_str': '▁game'},
 {'sequence': '[CLS] dont play games leh[SEP]',
  'score': 0.045703962445259094,
  'token': 466,
  'token_str': '▁games'},
 {'sequence': '[CLS] dont play around leh[SEP]',
  'score': 0.013458190485835075,
  'token': 140,
  'token_str': '▁around'}]

>>> nlp("catch no [MASK]")

[{'sequence': '[CLS] catch no ball[SEP]',
  'score': 0.6197211146354675,
  'token': 1592,
  'token_str': '▁ball'},
 {'sequence': '[CLS] catch no balls[SEP]',
  'score': 0.08441998809576035,
  'token': 7152,
  'token_str': '▁balls'},
 {'sequence': '[CLS] catch no joke[SEP]',
  'score': 0.0676785409450531,
  'token': 8186,
  'token_str': '▁joke'},
 {'sequence': '[CLS] catch no?[SEP]',
  'score': 0.040638409554958344,
  'token': 60,
  'token_str': '?'},
 {'sequence': '[CLS] catch no one[SEP]',
  'score': 0.03546864539384842,
  'token': 53,
  'token_str': '▁one'}]

>>> nlp("confirm plus [MASK]")

[{'sequence': '[CLS] confirm plus chop[SEP]',
  'score': 0.9608421921730042,
  'token': 17144,
  'token_str': '▁chop'},
 {'sequence': '[CLS] confirm plus guarantee[SEP]',
  'score': 0.011784233152866364,
  'token': 9120,
  'token_str': '▁guarantee'},
 {'sequence': '[CLS] confirm plus confirm[SEP]',
  'score': 0.010571340098977089,
  'token': 10265,
  'token_str': '▁confirm'},
 {'sequence': '[CLS] confirm plus egg[SEP]',
  'score': 0.0033525123726576567,
  'token': 6387,
  'token_str': '▁egg'},
 {'sequence': '[CLS] confirm plus bet[SEP]',
  'score': 0.0008760977652855217,
  'token': 5676,
  'token_str': '▁bet'}]

```

Here is how to use this model to get the features of a given text in PyTorch:
```python
from transformers import AlbertTokenizer, AlbertModel
tokenizer = AlbertTokenizer.from_pretrained('zanelim/singbert-lite-sg')
model = AlbertModel.from_pretrained("zanelim/singbert-lite-sg")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:
```python
from transformers import AlbertTokenizer, TFAlbertModel
tokenizer = AlbertTokenizer.from_pretrained("zanelim/singbert-lite-sg")
model = TFAlbertModel.from_pretrained("zanelim/singbert-lite-sg")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

#### Limitations and bias
This model was finetuned on colloquial Singlish and Manglish corpus, hence it is best applied on downstream tasks involving the main
constituent languages- english, mandarin, malay. Also, as the training data is mainly from forums, beware of existing inherent bias.

## Training data
Colloquial singlish and manglish (both are a mixture of English, Mandarin, Tamil, Malay, and other local dialects like Hokkien, Cantonese or Teochew)
corpus. The corpus is collected from subreddits- `r/singapore` and `r/malaysia`, and forums such as `hardwarezone`.

## Training procedure

Initialized with [albert base v2](https://github.com/google-research/albert#albert) vocab and checkpoints (pre-trained weights).

Pre-training was further finetuned on training data with the following hyperparameters
* train_batch_size: 4096
* max_seq_length: 128
* num_train_steps: 125000
* num_warmup_steps: 5000
* learning_rate: 0.00176
* hardware: TPU v3-8
