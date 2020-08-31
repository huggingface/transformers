---
language: en
tags:
- singapore
- sg
- singlish
- malaysia
- ms
- manglish
- bert-large-uncased
license: mit
datasets:
- reddit singapore, malaysia
- hardwarezone
widget:
- text: "kopi c siew [MASK]"
- text: "die [MASK] must try"
---

# Model name

SingBert Large - Bert for Singlish (SG) and Manglish (MY).

## Model description

Similar to [SingBert](https://huggingface.co/zanelim/singbert) but the large version, which was initialized from [BERT large uncased (whole word masking)](https://github.com/google-research/bert#pre-trained-models), with pre-training finetuned on
[singlish](https://en.wikipedia.org/wiki/Singlish) and [manglish](https://en.wikipedia.org/wiki/Manglish) data.

## Intended uses & limitations

#### How to use

```python
>>> from transformers import pipeline
>>> nlp = pipeline('fill-mask', model='zanelim/singbert-large-sg')
>>> nlp("kopi c siew [MASK]")

[{'sequence': '[CLS] kopi c siew dai [SEP]',
  'score': 0.9003700017929077,
  'token': 18765,
  'token_str': 'dai'},
 {'sequence': '[CLS] kopi c siew mai [SEP]',
  'score': 0.0779474675655365,
  'token': 14736,
  'token_str': 'mai'},
 {'sequence': '[CLS] kopi c siew. [SEP]',
  'score': 0.0032227332703769207,
  'token': 1012,
  'token_str': '.'},
 {'sequence': '[CLS] kopi c siew bao [SEP]',
  'score': 0.0017727474914863706,
  'token': 25945,
  'token_str': 'bao'},
 {'sequence': '[CLS] kopi c siew peng [SEP]',
  'score': 0.0012526646023616195,
  'token': 26473,
  'token_str': 'peng'}]

>>> nlp("one teh c siew dai, and one kopi [MASK]")

[{'sequence': '[CLS] one teh c siew dai, and one kopi. [SEP]',
  'score': 0.5249741077423096,
  'token': 1012,
  'token_str': '.'},
 {'sequence': '[CLS] one teh c siew dai, and one kopi o [SEP]',
  'score': 0.27349168062210083,
  'token': 1051,
  'token_str': 'o'},
 {'sequence': '[CLS] one teh c siew dai, and one kopi peng [SEP]',
  'score': 0.057190295308828354,
  'token': 26473,
  'token_str': 'peng'},
 {'sequence': '[CLS] one teh c siew dai, and one kopi c [SEP]',
  'score': 0.04022320732474327,
  'token': 1039,
  'token_str': 'c'},
 {'sequence': '[CLS] one teh c siew dai, and one kopi? [SEP]',
  'score': 0.01191170234233141,
  'token': 1029,
  'token_str': '?'}]

>>> nlp("die [MASK] must try")

[{'sequence': '[CLS] die die must try [SEP]',
  'score': 0.9921030402183533,
  'token': 3280,
  'token_str': 'die'},
 {'sequence': '[CLS] die also must try [SEP]',
  'score': 0.004993876442313194,
  'token': 2036,
  'token_str': 'also'},
 {'sequence': '[CLS] die liao must try [SEP]',
  'score': 0.000317625846946612,
  'token': 727,
  'token_str': 'liao'},
 {'sequence': '[CLS] die still must try [SEP]',
  'score': 0.0002260878391098231,
  'token': 2145,
  'token_str': 'still'},
 {'sequence': '[CLS] die i must try [SEP]',
  'score': 0.00016935862367972732,
  'token': 1045,
  'token_str': 'i'}]

>>> nlp("dont play [MASK] leh")

[{'sequence': '[CLS] dont play play leh [SEP]',
  'score': 0.9079819321632385,
  'token': 2377,
  'token_str': 'play'},
 {'sequence': '[CLS] dont play punk leh [SEP]',
  'score': 0.006846973206847906,
  'token': 7196,
  'token_str': 'punk'},
 {'sequence': '[CLS] dont play games leh [SEP]',
  'score': 0.004041737411171198,
  'token': 2399,
  'token_str': 'games'},
 {'sequence': '[CLS] dont play politics leh [SEP]',
  'score': 0.003728888463228941,
  'token': 4331,
  'token_str': 'politics'},
 {'sequence': '[CLS] dont play cheat leh [SEP]',
  'score': 0.0032805048394948244,
  'token': 21910,
  'token_str': 'cheat'}]

>>> nlp("confirm plus [MASK]")

{'sequence': '[CLS] confirm plus chop [SEP]',
  'score': 0.9749826192855835,
  'token': 24494,
  'token_str': 'chop'},
 {'sequence': '[CLS] confirm plus chopped [SEP]',
  'score': 0.017554156482219696,
  'token': 24881,
  'token_str': 'chopped'},
 {'sequence': '[CLS] confirm plus minus [SEP]',
  'score': 0.002725469646975398,
  'token': 15718,
  'token_str': 'minus'},
 {'sequence': '[CLS] confirm plus guarantee [SEP]',
  'score': 0.000900257145985961,
  'token': 11302,
  'token_str': 'guarantee'},
 {'sequence': '[CLS] confirm plus one [SEP]',
  'score': 0.0004384620988275856,
  'token': 2028,
  'token_str': 'one'}]

>>> nlp("catch no [MASK]")

[{'sequence': '[CLS] catch no ball [SEP]',
  'score': 0.9381157159805298,
  'token': 3608,
  'token_str': 'ball'},
 {'sequence': '[CLS] catch no balls [SEP]',
  'score': 0.060842301696538925,
  'token': 7395,
  'token_str': 'balls'},
 {'sequence': '[CLS] catch no fish [SEP]',
  'score': 0.00030917322146706283,
  'token': 3869,
  'token_str': 'fish'},
 {'sequence': '[CLS] catch no breath [SEP]',
  'score': 7.552534952992573e-05,
  'token': 3052,
  'token_str': 'breath'},
 {'sequence': '[CLS] catch no tail [SEP]',
  'score': 4.208395694149658e-05,
  'token': 5725,
  'token_str': 'tail'}]

```

Here is how to use this model to get the features of a given text in PyTorch:
```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('zanelim/singbert-large-sg')
model = BertModel.from_pretrained("zanelim/singbert-large-sg")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:
```python
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained("zanelim/singbert-large-sg")
model = TFBertModel.from_pretrained("zanelim/singbert-large-sg")
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

Initialized with [bert large uncased (whole word masking)](https://github.com/google-research/bert#pre-trained-models) vocab and checkpoints (pre-trained weights).
Top 1000 custom vocab tokens (non-overlapped with original bert vocab) were further extracted from training data and filled into unused tokens in original bert vocab.

Pre-training was further finetuned on training data with the following hyperparameters
* train_batch_size: 512
* max_seq_length: 128
* num_train_steps: 300000
* num_warmup_steps: 5000
* learning_rate: 2e-5
* hardware: TPU v3-8
