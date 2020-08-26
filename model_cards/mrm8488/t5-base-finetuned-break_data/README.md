---
language: en
datasets:
- break_data
---

# T5-base fine-tuned on break_data / QDMR-high-level ❓➡️📋

[Google's T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) fine-tuned on [break_data](https://huggingface.co/nlp/viewer/?dataset=break_data&config=QDMR-high-level) dataset for **QDMRs**.

## Details of T5 📜 ➡️ 📜

The **T5** model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu* in Here the abstract:

Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code.

![model image](https://i.imgur.com/jVFMMWR.png)


## Details of the downstream task (QDMRs) - Dataset 📚 

Break is a human annotated dataset of natural language questions and their Question Decomposition Meaning Representations (QDMRs). Break consists of 83,978 examples sampled from 10 question answering datasets over text, images and databases. This repository contains the Break dataset along with information on the exact data format. 

| Dataset  | Split | # samples |
| -------- | ----- | --------- |
| break_data | train | 17503    |
| break_data | valid  | 3130    |

Check out more about this dataset and others in [NLP Viewer](https://huggingface.co/nlp/viewer/)


## Model fine-tuning 🏋️‍
The training script is a slightly modified version of [this  awesome one](https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb) by [Suraj Patil](https://twitter.com/psuraj28). The main change is at preprocessing ```inputs``` and ```targets``` we feed to the model. We do it as a *paraphrasing task*.


## Model in Action 🚀

```python
# Tip: By now, install transformers from source

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-break_data")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-break_data")

def get_decomposition(question):
  input_text = "paraphrase: %s </s>" % question
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=32)

  return tokenizer.decode(output[0])

question = "The composer of Sands Theme plays what type of guitar?"

get_decomposition(question)

# output: 'return Sands Theme ;return composer of #1 ;return guitar that #2 plays'
```
> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
