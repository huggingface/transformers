---
language: "en"
tags:
- summarization
- extreme summarization
---
​
## Model description
​
BartForConditionalGenerationModel for extreme summarization- creates a one line abstractive summary of a given article
​
## How to use
​
PyTorch model available
​
```python
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
​
tokenizer = AutoTokenizer.from_pretrained("yuvraj/xSumm")			
model = AutoModelWithLMHead.from_pretrained("yuvraj/xSumm")
​
xsumm = pipeline('summarization', model=model, tokenizer=tokenizer)
xsumm("<text to be summarized>")
​
## Limitations and bias
Trained on a small fraction of the xsumm training dataset
