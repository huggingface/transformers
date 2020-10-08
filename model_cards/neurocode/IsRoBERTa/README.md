---
language: is
datasets:
- Icelandic portion of the OSCAR corpus from INRIA
- oscar
---

# IsRoBERTa a RoBERTa-like masked language model

Probably the first icelandic transformer language model!

## Overview
**Language:** Icelandic  
**Downstream-task:** masked-lm 
**Training data:** OSCAR corpus 
**Code:**  See [here](https://github.com/neurocode-io/icelandic-language-model)
**Infrastructure**: 1x Nvidia K80

## Hyperparameters

```
per_device_train_batch_size = 48
n_epochs = 1
vocab_size = 52.000
max_position_embeddings = 514
num_attention_heads = 12
num_hidden_layers = 6
type_vocab_size = 1
learning_rate=0.00005
``` 


## Usage

### In Transformers
```python
from transformers import (
  pipeline,
  AutoTokenizer,
  AutoModelWithLMHead
)

model_name = "neurocode/IsRoBERTa"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)
>>> fill_mask = pipeline(
...     "fill-mask",
...     model=model,
...     tokenizer=tokenizer
... )
>>> result = fill_mask("Hann fór út að <mask>.")
>>> result
[
  {'sequence': '<s>Hann fór út að nýju.</s>', 'score': 0.03395755589008331, 'token': 2219, 'token_str': 'ĠnÃ½ju'},
  {'sequence': '<s>Hann fór út að undanförnu.</s>', 'score': 0.029087543487548828, 'token': 7590, 'token_str': 'ĠundanfÃ¶rnu'},
  {'sequence': '<s>Hann fór út að lokum.</s>', 'score': 0.024420788511633873, 'token': 4384, 'token_str': 'Ġlokum'},
  {'sequence': '<s>Hann fór út að þessu.</s>', 'score': 0.021231256425380707, 'token': 921, 'token_str': 'ĠÃ¾essu'},
  {'sequence': '<s>Hann fór út að honum.</s>', 'score': 0.0205782949924469, 'token': 1136, 'token_str': 'Ġhonum'}
]
```


## Authors
Bobby Donchev: `contact [at] donchev.is`
Elena Cramer: `elena.cramer [at] neurocode.io`

## About us

We bring AI software for our customers live
Our focus: AI software development
 
Get in touch:
[LinkedIn](https://de.linkedin.com/company/neurocodeio) | [Website](https://neurocode.io)
