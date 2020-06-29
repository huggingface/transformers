---
language: english
datasets:
- squad_v2
---

# Longformer-base-4096 fine-tuned on SQuAD v2

[Longformer-base-4096 model](https://huggingface.co/allenai/longformer-base-4096) fine-tuned on [SQuAD v2](https://rajpurkar.github.io/SQuAD-explorer/) for **Q&A** downstream task.

## Longformer-base-4096

[Longformer](https://arxiv.org/abs/2004.05150) is a transformer model for long documents. 

`longformer-base-4096` is a BERT-like model started from the RoBERTa checkpoint and pretrained for MLM on long documents. It supports sequences of length up to 4,096. 
 
Longformer uses a combination of a sliding window (local) attention and global attention. Global attention is user-configured based on the task to allow the model to learn task-specific representations.

## Details of the downstream task (Q&A) - Dataset 📚 🧐 ❓

Dataset ID: ```squad_v2``` from  [HugginFace/NLP](https://github.com/huggingface/nlp)
| Dataset  | Split | # samples |
| -------- | ----- | --------- |
| squad_v2 | train | 130319      |
| squad_v2 | valid  | 11873     |

How to load it from [nlp](https://github.com/huggingface/nlp)

```python
train_dataset  = nlp.load_dataset('squad_v2', split=nlp.Split.TRAIN)
valid_dataset = nlp.load_dataset('squad_v2', split=nlp.Split.VALIDATION)
```
Check out more about this dataset and others in [NLP Viewer](https://huggingface.co/nlp/viewer/)


## Model fine-tuning 🏋️‍

The training script is a slightly modified version of [this one](https://colab.research.google.com/drive/1zEl5D-DdkBKva-DdreVOmN0hrAfzKG1o?usp=sharing)



## Model in Action 🚀

```python
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("mrm8488/longformer-base-4096-finetuned-squadv2")
model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/longformer-base-4096-finetuned-squadv2")

text = "Huggingface has democratized NLP. Huge thanks to Huggingface for this."
question = "What has Huggingface done ?"
encoding = tokenizer(question, text, return_tensors="pt")
input_ids = encoding["input_ids"]

# default is local attention everywhere
# the forward method will automatically set global attention on question tokens
attention_mask = encoding["attention_mask"]

start_scores, end_scores = model(input_ids, attention_mask=attention_mask)
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

# output => democratized NLP
```
If given the same context we ask something that is not there, the output for **no answer** will be ```<s>```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
