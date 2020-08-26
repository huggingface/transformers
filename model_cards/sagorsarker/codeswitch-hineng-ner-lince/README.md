---
language:
- hi
- en
---

# codeswitch-hineng-ner-lince
This is a pretrained model for **Name Entity Recognition** of `Hindi-english` code-mixed data used from [LinCE](https://ritual.uh.edu/lince/home)

This model is trained for this below repository. 

[https://github.com/sagorbrur/codeswitch](https://github.com/sagorbrur/codeswitch)

To install codeswitch:

```
pip install codeswitch
```

## Identify Language

* Method-1

```py

from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-hineng-ner-lince")

model = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-hineng-ner-lince")

ner_model = pipeline('ner', model=model, tokenizer=tokenizer)

ner_model("put any hindi english code-mixed sentence")

```

* Method-2

```py
from codeswitch.codeswitch import NER
ner = NER('hin-eng')
text = "" # your mixed sentence 
result = ner.tag(text)
print(result)

```
