---
language: english
thumbnail:
---

# T5-base fine-tuned on SQuAD v2

[Google's T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) fine-tuned on [SQuAD v2](https://rajpurkar.github.io/SQuAD-explorer/) for **Q&A** downstream task.

## Details of T5

The **T5** model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu* in Here the abstract:

Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code.


## Details of the downstream task (Q&A) - Dataset 📚 🧐 ❓

[SQuAD v2](https://rajpurkar.github.io/SQuAD-explorer/) combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

| Dataset  | Split | # samples |
| -------- | ----- | --------- |
| SQuAD2.0 | train | 130k      |
| SQuAD2.0 | eval  | 12.3k     |



## Model fine-tuning 🏋️‍

The training script is a slightly modified version of [this one](https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb)

## Results 📝

| Metric | # Value   |
| ------ | --------- |
| **EM** | **77.64** |
| **F1** | **81.32** |



## Model in Action 🚀

```python
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-squadv2")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-squadv2")

def get_answer(question, context):
  input_text = "question: %s  context: %s </s>" % (question, context)
  features = tokenizer.batch_encode_plus([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'])
  
  return tokenizer.decode(output[0])

context = "Manuel have created RuPERTa-base with the support of HF-Transformers and Google"
question = "Who has supported Manuel?"

get_answer(question, context)

# output: 'HF-Transformers and Google'
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
