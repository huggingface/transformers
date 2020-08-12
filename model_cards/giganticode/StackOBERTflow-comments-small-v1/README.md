# StackOBERTflow-comments-small

StackOBERTflow is a RoBERTa model trained on StackOverflow comments.
A Byte-level BPE tokenizer with dropout was used (using the `tokenizers` package).

The model is *small*, i.e. has only 6-layers and the maximum sequence length was restricted to 256 tokens. 
The model was trained for 6 epochs on several GBs of comments from the StackOverflow corpus.

## Quick start: masked language modeling prediction

```python
from transformers import pipeline
from pprint import pprint

COMMENT = "You really should not do it this way, I would use <mask> instead."

fill_mask = pipeline(
    "fill-mask",
    model="giganticode/StackOBERTflow-comments-small-v1",
    tokenizer="giganticode/StackOBERTflow-comments-small-v1"
)

pprint(fill_mask(COMMENT))
# [{'score': 0.019997311756014824,
#   'sequence': '<s> You really should not do it this way, I would use jQuery instead.</s>',
#   'token': 1738},
#  {'score': 0.01693696901202202,
#   'sequence': '<s> You really should not do it this way, I would use arrays instead.</s>',
#   'token': 2844},
#  {'score': 0.013411642983555794,
#   'sequence': '<s> You really should not do it this way, I would use CSS instead.</s>',
#   'token': 2254},
#  {'score': 0.013224546797573566,
#   'sequence': '<s> You really should not do it this way, I would use it instead.</s>',
#   'token': 300},
#  {'score': 0.011984303593635559,
#   'sequence': '<s> You really should not do it this way, I would use classes instead.</s>',
#   'token': 1779}]
```
