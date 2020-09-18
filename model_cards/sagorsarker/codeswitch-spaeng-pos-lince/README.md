---
language:
- es
- en
datasets:
- lince
license: "MIT"
tags:
- codeswitching
- spanish-english
- pos
---

# codeswitch-spaeng-pos-lince
This is a pretrained model for **Part of Speech Tagging** of `spanish-english` code-mixed data used from [LinCE](https://ritual.uh.edu/lince/home)

This model is trained for this below repository. 

[https://github.com/sagorbrur/codeswitch](https://github.com/sagorbrur/codeswitch)

To install codeswitch:

```
pip install codeswitch
```

## Part-of-Speech Tagging of Spanish-English Mixed Data

* **Method-1**

```py

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-spaeng-pos-lince")

model = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-spaeng-pos-lince")
pos_model = pipeline('ner', model=model, tokenizer=tokenizer)

pos_model("put any spanish english code-mixed sentence")

```

* **Method-2**

```py
from codeswitch.codeswitch import POS
pos = POS('spa-eng')
text = "" # your mixed sentence 
result = pos.tag(text)
print(result)
```
