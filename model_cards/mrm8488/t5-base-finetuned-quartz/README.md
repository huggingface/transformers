---
language: en
datasets:
- quartz
pipeline_tag: question-answering
---

# T5-base fine-tuned on QuaRTz  

[Google's T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) fine-tuned on [QuaRTz](https://allenai.org/data/quartz) for **QA** downstream task.

## Details of T5

The **T5** model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu* in Here the abstract:

Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code.

![model image](https://i.imgur.com/jVFMMWR.png)


## Details of the dataset 📚 

**QuaRTz** is a crowdsourced dataset of 3864 multiple-choice questions about open domain qualitative relationships. Each question is paired with one of 405 different background sentences (sometimes short paragraphs). The QuaRTz dataset V1 contains 3864 questions about open domain qualitative relationships. Each question is paired with one of 405 different background sentences (sometimes short paragraphs).
The dataset is split into:

|Set  | Samples|
|-----|--------|
|Train | 2696 |
|Valid | 384 |
|Test | 784 |

## Model fine-tuning 🏋️‍

The training script is a slightly modified version of [this  awesome one](https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb) by [Suraj Patil](https://twitter.com/psuraj28). The *question*, *context* (`para` field) and *options* (`choices` field) are concatenated and passed to the **encoder**. The **decoder** receives the right *answer* (by querying `answerKey` field). More details about the dataset fields/format [here](https://huggingface.co/nlp/viewer/?dataset=quartz) 

## Results 📋 


|Set   | Metric | Score |
|-----|--------|-------|
|Validation | Accuracy (EM) | **83.59**|
|Test | Accuracy (EM) | **81.50**|


## Model in Action 🚀

```python
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-quartz")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-quartz")

def get_response(question, fact, opts, max_length=16):
  input_text = 'question: %s context: %s options: %s' % (question, fact, opts)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0])
  
fact = 'The sooner cancer is detected the easier it is to treat.'
question = 'John was a doctor in a cancer ward and knew that early detection was key. The cancer being detected quickly makes the cancer treatment'
opts = 'Easier, Harder'

get_response(question, fact, opts)

# output: 'Easier'
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
