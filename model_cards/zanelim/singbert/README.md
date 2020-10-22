---
language: en
tags:
- singapore
- sg
- singlish
- malaysia
- ms
- manglish
- bert-base-uncased
license: mit
datasets:
- reddit singapore, malaysia
- hardwarezone
widget:
- text: "kopi c siew [MASK]"
- text: "die [MASK] must try"
---

# Model name

SingBert - Bert for Singlish (SG) and Manglish (MY).

## Model description

[BERT base uncased](https://github.com/google-research/bert#pre-trained-models), with pre-training finetuned on
[singlish](https://en.wikipedia.org/wiki/Singlish) and [manglish](https://en.wikipedia.org/wiki/Manglish) data.

## Intended uses & limitations

#### How to use

```python
>>> from transformers import pipeline
>>> nlp = pipeline('fill-mask', model='zanelim/singbert')
>>> nlp("kopi c siew [MASK]")

[{'sequence': '[CLS] kopi c siew dai [SEP]',
  'score': 0.5092713236808777,
  'token': 18765,
  'token_str': 'dai'},
 {'sequence': '[CLS] kopi c siew mai [SEP]',
  'score': 0.3515934646129608,
  'token': 14736,
  'token_str': 'mai'},
 {'sequence': '[CLS] kopi c siew bao [SEP]',
  'score': 0.05576375499367714,
  'token': 25945,
  'token_str': 'bao'},
 {'sequence': '[CLS] kopi c siew. [SEP]',
  'score': 0.006019321270287037,
  'token': 1012,
  'token_str': '.'},
 {'sequence': '[CLS] kopi c siew sai [SEP]',
  'score': 0.0038361591286957264,
  'token': 18952,
  'token_str': 'sai'}]

>>> nlp("one teh c siew dai, and one kopi [MASK].")

[{'sequence': '[CLS] one teh c siew dai, and one kopi c [SEP]',
  'score': 0.6176503300666809,
  'token': 1039,
  'token_str': 'c'},
 {'sequence': '[CLS] one teh c siew dai, and one kopi o [SEP]',
  'score': 0.21094971895217896,
  'token': 1051,
  'token_str': 'o'},
 {'sequence': '[CLS] one teh c siew dai, and one kopi. [SEP]',
  'score': 0.13027705252170563,
  'token': 1012,
  'token_str': '.'},
 {'sequence': '[CLS] one teh c siew dai, and one kopi! [SEP]',
  'score': 0.004680239595472813,
  'token': 999,
  'token_str': '!'},
 {'sequence': '[CLS] one teh c siew dai, and one kopi w [SEP]',
  'score': 0.002034128177911043,
  'token': 1059,
  'token_str': 'w'}]

>>> nlp("dont play [MASK] leh")

[{'sequence': '[CLS] dont play play leh [SEP]',
  'score': 0.9281464219093323,
  'token': 2377,
  'token_str': 'play'},
 {'sequence': '[CLS] dont play politics leh [SEP]',
  'score': 0.010990909300744534,
  'token': 4331,
  'token_str': 'politics'},
 {'sequence': '[CLS] dont play punk leh [SEP]',
  'score': 0.005583590362221003,
  'token': 7196,
  'token_str': 'punk'},
 {'sequence': '[CLS] dont play dirty leh [SEP]',
  'score': 0.0025784350000321865,
  'token': 6530,
  'token_str': 'dirty'},
 {'sequence': '[CLS] dont play cheat leh [SEP]',
  'score': 0.0025066907983273268,
  'token': 21910,
  'token_str': 'cheat'}]

>>> nlp("catch no [MASK]")

[{'sequence': '[CLS] catch no ball [SEP]',
  'score': 0.7922210693359375,
  'token': 3608,
  'token_str': 'ball'},
 {'sequence': '[CLS] catch no balls [SEP]',
  'score': 0.20503675937652588,
  'token': 7395,
  'token_str': 'balls'},
 {'sequence': '[CLS] catch no tail [SEP]',
  'score': 0.0006608376861549914,
  'token': 5725,
  'token_str': 'tail'},
 {'sequence': '[CLS] catch no talent [SEP]',
  'score': 0.0002158183924620971,
  'token': 5848,
  'token_str': 'talent'},
 {'sequence': '[CLS] catch no prisoners [SEP]',
  'score': 5.3481446229852736e-05,
  'token': 5895,
  'token_str': 'prisoners'}]

>>> nlp("confirm plus [MASK]")

[{'sequence': '[CLS] confirm plus chop [SEP]',
  'score': 0.992355227470398,
  'token': 24494,
  'token_str': 'chop'},
 {'sequence': '[CLS] confirm plus one [SEP]',
  'score': 0.0037301010452210903,
  'token': 2028,
  'token_str': 'one'},
 {'sequence': '[CLS] confirm plus minus [SEP]',
  'score': 0.0014284878270700574,
  'token': 15718,
  'token_str': 'minus'},
 {'sequence': '[CLS] confirm plus 1 [SEP]',
  'score': 0.0011354683665558696,
  'token': 1015,
  'token_str': '1'},
 {'sequence': '[CLS] confirm plus chopped [SEP]',
  'score': 0.0003804611915256828,
  'token': 24881,
  'token_str': 'chopped'}]

>>> nlp("die [MASK] must try")

[{'sequence': '[CLS] die die must try [SEP]',
  'score': 0.9552758932113647,
  'token': 3280,
  'token_str': 'die'},
 {'sequence': '[CLS] die also must try [SEP]',
  'score': 0.03644804656505585,
  'token': 2036,
  'token_str': 'also'},
 {'sequence': '[CLS] die liao must try [SEP]',
  'score': 0.003282855963334441,
  'token': 727,
  'token_str': 'liao'},
 {'sequence': '[CLS] die already must try [SEP]',
  'score': 0.0004937972989864647,
  'token': 2525,
  'token_str': 'already'},
 {'sequence': '[CLS] die hard must try [SEP]',
  'score': 0.0003659659414552152,
  'token': 2524,
  'token_str': 'hard'}]

```

Here is how to use this model to get the features of a given text in PyTorch:
```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('zanelim/singbert')
model = BertModel.from_pretrained("zanelim/singbert")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:
```python
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained("zanelim/singbert")
model = TFBertModel.from_pretrained("zanelim/singbert")
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

Initialized with [bert base uncased](https://github.com/google-research/bert#pre-trained-models) vocab and checkpoints (pre-trained weights).
Top 1000 custom vocab tokens (non-overlapped with original bert vocab) were further extracted from training data and filled into unused tokens in original bert vocab.

Pre-training was further finetuned on training data with the following hyperparameters
* train_batch_size: 512
* max_seq_length: 128
* num_train_steps: 300000
* num_warmup_steps: 5000
* learning_rate: 2e-5
* hardware: TPU v3-8
