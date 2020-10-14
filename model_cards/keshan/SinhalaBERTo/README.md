---
language: si
tags:
- SinhalaBERTo
- Sinhala
- roberta
datasets:
- oscar
---
### Overview

This is a slightly smaller model trained on [OSCAR](https://oscar-corpus.com/) Sinhala dedup dataset. As Sinhala is one of those low resource languages, there are only a handful of models been trained. So, this would be a great place to start training for more downstream tasks. 

## Model Specification


The model chosen for training is [Roberta](https://arxiv.org/abs/1907.11692) with the following specifications:
 1. vocab_size=52000
 2. max_position_embeddings=514
 3. num_attention_heads=12
 4. num_hidden_layers=6
 5. type_vocab_size=1
 
## How to Use
You can use this model directly with a pipeline for masked language modeling:

```py
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline

model = BertForMaskedLM.from_pretrained("keshan/SinhalaBERTo")
tokenizer = BertTokenizer.from_pretrained("keshan/SinhalaBERTo")

fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

fill_mask("මම ගෙදර <mask>.")

```
