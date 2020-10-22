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
- sentiment-analysis
---

# codeswitch-spaeng-sentiment-analysis-lince
This is a pretrained model for **Sentiment Analysis** of `spanish-english` code-mixed data used from [LinCE](https://ritual.uh.edu/lince/home)

This model is trained for this below repository. 

[https://github.com/sagorbrur/codeswitch](https://github.com/sagorbrur/codeswitch)

To install codeswitch:

```
pip install codeswitch
```

## Sentiment Analysis of Spanish-English  Code-Mixed Data

* **Method-1**

```py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-spaeng-sentiment-analysis-lince")

model = AutoModelForSequenceClassification.from_pretrained("sagorsarker/codeswitch-spaeng-sentiment-analysis-lince")

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
sentence = "El perro le ladraba a La Gatita .. .. lol #teamlagatita en las playas de Key Biscayne este Memorial day"
nlp(sentence)

```

* **Method-2**

```py
from codeswitch.codeswitch import SentimentAnalysis
sa = SentimentAnalysis('spa-eng')
sentence = "El perro le ladraba a La Gatita .. .. lol #teamlagatita en las playas de Key Biscayne este Memorial day"
result = sa.analyze(sentence)
print(result)
```
