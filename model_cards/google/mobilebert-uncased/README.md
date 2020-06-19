---
language: english
thumbnail: https://huggingface.co/front/thumbnails/google.png

license: apache-2.0
---

## MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices

MobileBERT is a thin version of BERT_LARGE, while equipped with bottleneck structures and a carefully designed balance
between self-attentions and feed-forward networks.

This checkpoint is the original MobileBert Optimized Uncased English: 
[uncased_L-24_H-128_B-512_A-4_F-4_OPT](https://storage.googleapis.com/cloud-tpu-checkpoints/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT.tar.gz) 
checkpoint.

## How to use MobileBERT in `transformers`

```python
from transformers import pipeline

fill_mask = pipeline(
	"fill-mask",
	model="google/mobilebert-uncased",
	tokenizer="google/mobilebert-uncased"
)

print(
	fill_mask(f"HuggingFace is creating a {fill_mask.tokenizer.mask_token} that the community uses to solve NLP tasks.")
)

```
