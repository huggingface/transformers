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
- ner
---

# codeswitch-spaeng-ner-lince
This is a pretrained model for **Name Entity Recognition** of `spanish-english` code-mixed data used from [LinCE](https://ritual.uh.edu/lince/home)

This model is trained for this below repository. 

[https://github.com/sagorbrur/codeswitch](https://github.com/sagorbrur/codeswitch)

To install codeswitch:

```
pip install codeswitch
```

## Name Entity Recognition of Spanish-English Mixed Data

* **Method-1**

```py

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-spaeng-ner-lince")

model = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-spaeng-ner-lince")

ner_model = pipeline('ner', model=model, tokenizer=tokenizer)

ner_model("put any spanish english code-mixed sentence")

```

* **Method-2**

```py
from codeswitch.codeswitch import NER
ner = NER('spa-eng')
text = "" # your mixed sentence 
result = ner.tag(text)
print(result)
```
