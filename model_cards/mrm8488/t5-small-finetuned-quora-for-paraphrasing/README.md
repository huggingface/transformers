---
language: en
datasets:
- quora
---

# T5-base fine-tuned on Quora question pair dataset for Question Paraphrasing ❓↔️❓

[Google's T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) fine-tuned on [Quodra question pair](https://huggingface.co/nlp/viewer/?dataset=quora) dataset for **Question Paraphrasing** task.

## Details of T5

The **T5** model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu* in Here the abstract:

Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code.

![model image](https://i.imgur.com/jVFMMWR.png)


## Details of the downstream task (Question Paraphrasing) - Dataset 📚❓↔️❓

Dataset ID: ```quora``` from  [Huggingface/NLP](https://github.com/huggingface/nlp)

| Dataset  | Split | # samples |
| -------- | ----- | --------- |
| quora | train | 404290    |
| quora after filter repeated questions | train  | 149263    |

Check out more about this dataset and others in [NLP Viewer](https://huggingface.co/nlp/viewer/)


## Model fine-tuning 🏋️‍

The training script is a slightly modified version of [this one](https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb)



## Model in Action 🚀

```python
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")

def paraphrase(text, max_length=128):

  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

  generated_ids = model.generate(input_ids=input_ids, num_return_sequences=5, num_beams=5, max_length=max_length, no_repeat_ngram_size=2, repetition_penalty=3.5, length_penalty=1.0, early_stopping=True)

  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

  return preds
  
preds = paraphrase("paraphrase: What is the best framework for dealing with a huge text dataset?")

for pred in preds:
  print(pred)

# Output:
'''
What is the best framework for dealing with a huge text dataset?
What is the best framework for dealing with a large text dataset?
What is the best framework to deal with a huge text dataset?
What are the best frameworks for dealing with a huge text dataset?
What is the best framework for dealing with huge text datasets?
'''
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
