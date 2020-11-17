---
language: en
datasets:
- codexglue
---

# CodeBERT fine-tuned for Insecure Code Detection 💾⛔


[codebert-base](https://huggingface.co/microsoft/codebert-base) fine-tuned on [CodeXGLUE -- Defect Detection](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection) dataset for **Insecure Code Detection** downstream task.

## Details of [CodeBERT](https://arxiv.org/abs/2002.08155)

We present CodeBERT, a bimodal pre-trained model for programming language (PL) and nat-ural language (NL). CodeBERT learns general-purpose representations that support downstream NL-PL applications such as natural language codesearch, code documentation generation, etc. We develop CodeBERT with Transformer-based neural architecture, and train it with a hybrid objective function that incorporates the pre-training task of replaced token detection, which is to detect plausible alternatives sampled from generators. This enables us to utilize both bimodal data of NL-PL pairs and unimodal data, where the former provides input tokens for model training while the latter helps to learn better generators. We evaluate CodeBERT on two NL-PL applications by fine-tuning model parameters. Results show that CodeBERT achieves state-of-the-art performance on both natural language code search and code documentation generation tasks. Furthermore, to investigate what type of knowledge is learned in CodeBERT, we construct a dataset for NL-PL probing, and evaluate in a zero-shot setting where parameters of pre-trained models are fixed. Results show that CodeBERT performs better than previous pre-trained models on NL-PL probing.

## Details of the downstream task (code classification) - Dataset 📚

Given a source code, the task is to identify whether it is an insecure code that may attack software systems, such as resource leaks, use-after-free vulnerabilities and DoS attack.  We treat the task as binary classification (0/1), where 1 stands for insecure code and 0 for secure code.

The [dataset](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection) used comes from the paper [*Devign*: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks](http://papers.nips.cc/paper/9209-devign-effective-vulnerability-identification-by-learning-comprehensive-program-semantics-via-graph-neural-networks.pdf). All projects are combined and splitted 80%/10%/10% for training/dev/test.

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  21,854   |
| Dev   |   2,732   |
| Test  |   2,732   |

## Test set metrics 🧾

| Methods  |    ACC    |
| -------- | :-------: |
| BiLSTM   |   59.37   |
| TextCNN  |   60.69   |
| [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)  |   61.05   |
| [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) | 62.08 |
| [Ours](https://huggingface.co/mrm8488/codebert-base-finetuned-detect-insecure-code)  | **65.30** |


## Model in Action 🚀

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
tokenizer = AutoTokenizer.from_pretrained('mrm8488/codebert-base-finetuned-detect-insecure-code')
model = AutoModelForSequenceClassification.from_pretrained('mrm8488/codebert-base-finetuned-detect-insecure-code')

inputs = tokenizer("your code here", return_tensors="pt", truncation=True, padding='max_length')
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

print(np.argmax(logits.detach().numpy()))
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
