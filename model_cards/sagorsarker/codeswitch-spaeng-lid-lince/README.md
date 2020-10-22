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
- language-identification
---

# codeswitch-spaeng-lid-lince
This is a pretrained model for **language identification** of `spanish-english` code-mixed data used from [LinCE](https://ritual.uh.edu/lince/home)

This model is trained for this below repository. 

[https://github.com/sagorbrur/codeswitch](https://github.com/sagorbrur/codeswitch)

To install codeswitch:

```
pip install codeswitch
```

## Identify Language

* **Method-1**

```py

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-spaeng-lid-lince")

model = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-spaeng-lid-lince")
lid_model = pipeline('ner', model=model, tokenizer=tokenizer)

lid_model("put any spanish english code-mixed sentence")

```

* **Method-2**

```py
from codeswitch.codeswitch import LanguageIdentification
lid = LanguageIdentification('spa-eng') 
text = "" # your code-mixed sentence 
result = lid.identify(text)
print(result)
```
