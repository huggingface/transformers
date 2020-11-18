---
language: 
- ach 
- en
tags:
- translation
license: cc-by-4.0
datasets:
- JW300
metrics:
- bleu
---

# HEL-ACH-EN

## Model description

MT model translating Acholi to English initialized with weights from [opus-mt-luo-en](https://huggingface.co/Helsinki-NLP/opus-mt-luo-en) on HuggingFace.

## Intended uses & limitations
Machine Translation experiments. Do not use for sensitive tasks.
#### How to use

```python
# You can include sample code which will be formatted
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Ogayo/Hel-ach-en")

model = AutoModelForSeq2SeqLM.from_pretrained("Ogayo/Hel-ach-en")

```

#### Limitations and bias

Trained on Jehovah Witnesses data so contains theirs and Christian views.

## Training data
Trained on OPUS JW300 data.
Initialized with weights from [opus-mt-luo-en](https://huggingface.co/Helsinki-NLP/opus-mt-luo-en?text=Bed+gi+nyasi+mar+chieng%27+nyuol+mopong%27+gi+mor%21#model_card)

## Training procedure

Remove duplicates and rows with no alphabetic characters. Used GPU
## Eval results
testset | BLEU 
--- | --- 
JW300.luo.en| 46.1
