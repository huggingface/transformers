---
language: zh-tw
---

# Model name
Chinese-bert-wwm-electrical-health-record-ner-sequence-labeling


#### How to use

```
from transformers import AutoTokenizer, AutoModelForTokenClassification  
tokenizer = AutoTokenizer.from_pretrained("chinese-bert-wwm-ehr-ner-sl")  
model = AutoModelForTokenClassification.from_pretrained("chinese-bert-wwm-ehr-ner-sl") 
```
