---
language: "en"
tags:
- summarization
---
​
# Summarization
​
## Model description
​
BartForConditionalGeneration model fine tuned for summarization on 10000 samples from the cnn-dailymail dataset
​
## How to use
​
PyTorch model available
​
```python
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
​
tokenizer = AutoTokenizer.from_pretrained("yuvraj/summarizer-cnndm") 
AutoModelWithLMHead.from_pretrained("yuvraj/summarizer-cnndm")
​
summarizer = pipeline('summarization', model=model, tokenizer=tokenizer)
summarizer("<Text to be summarized>")
​
## Limitations and bias
Trained on a small dataset
