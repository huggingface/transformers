---
language:
- hi
- en
---

# codeswitch-hineng-lid-lince
This is a pretrained model for **language identification** of `hindi-english` code-mixed data used from [LinCE](https://ritual.uh.edu/lince/home)

## Identify Language

* Method-1

```py

from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-hineng-lid-lince")

model = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-hineng-lid-lince")
lid_model = pipeline('ner', model=model, tokenizer=tokenizer)

lid_model("put any hindi english code-mixed sentence")

```

* Method-2

```py
# !pip install codeswitch
from codeswitch.codeswitch import LanguageIdentification
lid = LanguageIdentification('hin-eng') 
text = "" # your code-mixed sentence 
result = lid.identify(text)
print(result)
```

