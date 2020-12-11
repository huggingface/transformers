---
language: en
datasets:
- wikisql
---

# T5-small fine-tuned on WikiSQL

[Google's T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) [small](https://huggingface.co/t5-small) fine-tuned on [WikiSQL](https://github.com/salesforce/WikiSQL) for **English** to **SQL** **translation**.

## Details of T5

The **T5** model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu* in Here the abstract:

Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code.

![model image](https://i.imgur.com/jVFMMWR.png)


## Details of the Dataset 📚 

Dataset ID: ```wikisql``` from  [Huggingface/NLP](https://huggingface.co/nlp/viewer/?dataset=wikisql)

| Dataset  | Split | # samples |
| -------- | ----- | --------- |
| wikisql | train | 56355    |
| wikisql | valid  | 14436    |

How to load it from [nlp](https://github.com/huggingface/nlp)

```python
train_dataset  = nlp.load_dataset('wikisql', split=nlp.Split.TRAIN)
valid_dataset = nlp.load_dataset('wikisql', split=nlp.Split.VALIDATION)
```
Check out more about this dataset and others in [NLP Viewer](https://huggingface.co/nlp/viewer/)


## Model fine-tuning 🏋️‍

The training script is a slightly modified version of [this Colab Notebook](https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb) created by [Suraj Patil](https://github.com/patil-suraj), so all credits to him!



## Model in Action 🚀

```python
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-small-finetuned-wikiSQL")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-small-finetuned-wikiSQL")

def get_sql(query):
  input_text = "translate English to SQL: %s </s>" % query
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'])
  
  return tokenizer.decode(output[0])

query = "How many millions of params there are in HF-hub?"

get_sql(query)

# output: 'SELECT COUNT Params FROM table WHERE Location = HF-hub'
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
