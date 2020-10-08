---
language: 
- sot
thumbnail: https://pbs.twimg.com/media/EVjR6BsWoAAFaq5.jpg
tags:
- sot
- fill-mask
- pytorch
- roberta
- lm-head
- masked-lm
license: MIT
---

# Takalani Sesame - Southern Sotho 🇿🇦

<img src="https://pbs.twimg.com/media/EVjR6BsWoAAFaq5.jpg" width="600"/> 

## Model description

Takalani Sesame (named after the South African version of Sesame Street) is a project that aims to promote the use of South African languages in NLP, and in particular look at techniques for low-resource languages to equalise performance with larger languages around the world.

## Intended uses & limitations

#### How to use

```python
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("jannesg/takalane_sot_roberta")

model = AutoModelWithLMHead.from_pretrained("jannesg/takalane_sot_roberta")
```

#### Limitations and bias

Updates will be added continously to improve performance. 

## Training data

Data collected from [https://wortschatz.uni-leipzig.de/en](https://wortschatz.uni-leipzig.de/en) <br/>
**Sentences:** 20000

## Training procedure

No preprocessing. Standard Huggingface hyperparameters. 

## Author

Jannes Germishuys [website](http://jannesgg.github.io)
