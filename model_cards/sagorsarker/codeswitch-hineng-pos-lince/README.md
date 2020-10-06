---
language:
- hi
- en
datasets:
- lince
license: "MIT"
tags:
- codeswitching
- hindi-english
- pos
---

# codeswitch-hineng-pos-lince
This is a pretrained model for **Part of Speech Tagging** of `hindi-english` code-mixed data used from [LinCE](https://ritual.uh.edu/lince/home)

This model is trained for this below repository. 

[https://github.com/sagorbrur/codeswitch](https://github.com/sagorbrur/codeswitch)

To install codeswitch:

```
pip install codeswitch
```

## Part-of-Speech Tagging of Hindi-English Mixed Data

* **Method-1**

```py

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-hineng-pos-lince")

model = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-hineng-pos-lince")
pos_model = pipeline('ner', model=model, tokenizer=tokenizer)

pos_model("put any hindi english code-mixed sentence")

```

* **Method-2**

```py
from codeswitch.codeswitch import POS
pos = POS('hin-eng')
text = "" # your mixed sentence 
result = pos.tag(text)
print(result)
```
