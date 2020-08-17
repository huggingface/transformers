---
language: ur
thumbnail: https://raw.githubusercontent.com/urduhack/urduhack/master/docs/_static/urduhack.png
tags:
- roberta-urdu-small
- urdu
- transformers
license: mit
---
## roberta-urdu-small

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/urduhack/urduhack/blob/master/LICENSE)
### Overview
**Language model:** roberta-urdu-small
**Model size:** 125M
**Language:** Urdu
**Training data:** News data from urdu news resources in Pakistan
### About roberta-urdu-small
roberta-urdu-small is a language model for urdu language.
```
from transformers import pipeline
fill_mask = pipeline("fill-mask", model="urduhack/roberta-urdu-small", tokenizer="urduhack/roberta-urdu-small")
```
## Training procedure
roberta-urdu-small was trained on urdu news corpus. Training data was normalized using normalization module from
urduhack to eliminate characters from other languages like arabic.

### About Urduhack
Urduhack is a Natural Language Processing (NLP) library for urdu language.
Github: https://github.com/urduhack/urduhack
