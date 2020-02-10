---
tags:
- ci
---

## Dummy model used for unit testing and CI


```python
import json
import os
from transformers.configuration_roberta import RobertaConfig
from transformers import RobertaForMaskedLM, TFRobertaForMaskedLM

DIRNAME = "./dummy-unknown"


config = RobertaConfig(10, 20, 1, 1, 40)

model = RobertaForMaskedLM(config)
model.save_pretrained(DIRNAME)

tf_model = TFRobertaForMaskedLM.from_pretrained(DIRNAME, from_pt=True)
tf_model.save_pretrained(DIRNAME)

# Tokenizer:

vocab = [
    "l",
    "o",
    "w",
    "e",
    "r",
    "s",
    "t",
    "i",
    "d",
    "n",
    "\u0120",
    "\u0120l",
    "\u0120n",
    "\u0120lo",
    "\u0120low",
    "er",
    "\u0120lowest",
    "\u0120newer",
    "\u0120wider",
    "<unk>",
]
vocab_tokens = dict(zip(vocab, range(len(vocab))))
merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]

vocab_file = os.path.join(DIRNAME, "vocab.json")
merges_file = os.path.join(DIRNAME, "merges.txt")
with open(vocab_file, "w", encoding="utf-8") as fp:
    fp.write(json.dumps(vocab_tokens) + "\n")
with open(merges_file, "w", encoding="utf-8") as fp:
    fp.write("\n".join(merges))
```
