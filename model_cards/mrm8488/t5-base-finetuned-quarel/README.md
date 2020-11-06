---
language: en
datasets:
- quarel
---

# T5-base fine-tuned on QuaRel 

[Google's T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) fine-tuned on [QuaRel](https://allenai.org/data/quarel) for **QA** downstream task.

## Details of T5

The **T5** model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu* in Here the abstract:

Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code.

![model image](https://i.imgur.com/jVFMMWR.png)


## Details of the dataset 📚 

**QuaRel**: *[A Dataset and Models for Answering Questions about Qualitative Relationships](https://www.semanticscholar.org/paper/QuaRel%3A-A-Dataset-and-Models-for-Answering-about-Tafjord-Clark/51004bc6461a572e1189a0e3b32b441155d760ce)*

Many natural language questions require recognizing and reasoning with qualitative relationships (e.g., in science, economics, and medicine), but are challenging to answer with corpus-based methods. Qualitative modeling provides tools that support such reasoning, but the semantic parsing task of mapping questions into those models has formidable challenges. We present QuaRel, a dataset of diverse story questions involving qualitative relationships that characterize these challenges, and techniques that begin to address them. The dataset has 2771 questions relating 19 different types of quantities. For example, "Jenny observes that the robot vacuum cleaner moves slower on the living room carpet than on the bedroom carpet. Which carpet has more friction?" We contribute (1) a simple and flexible conceptual framework for representing these kinds of questions; (2) the QuaRel dataset, including logical forms, exemplifying the parsing challenges; and (3) two novel models for this task, built as extensions of type-constrained semantic parsing. The first of these models (called QuaSP+) significantly outperforms off-the-shelf tools on QuaRel. The second (QuaSP+Zero) demonstrates zero-shot capability, i.e., the ability to handle new qualitative relationships without requiring additional training data, something not possible with previous models. This work thus makes inroads into answering complex, qualitative questions that require reasoning, and scaling to new relationships at low cost

## Model fine-tuning 🏋️‍

The training script is a slightly modified version of [this  awesome one](https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb) by [Suraj Patil](https://twitter.com/psuraj28). The **context** passed to the *encoder* is the `logical_form_pretty` field (example: `qrel(speed, higher, ice) -> qrel(smoothness, higher, snow) ; qrel(smoothness, higher, ice`) . The **question** is just the `question` field. The **answer** passed to the *decoder* is obtained from `question`using the `answer_index` field. More details about the dataset format/fields [here](https://huggingface.co/nlp/viewer/?dataset=quarel)

## Metrics on validation set 📋 

| Metric | Score |
|--------|-------|
|Accuracy (EM) | **67.98**|


## Model in Action 🚀

```python
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-quarel")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-quarel")

def get_response(question, context, max_length=32):
  input_text = 'question: %s  context: %s' % (question, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0])
  
question = 'As the train left the station it crossed the bridge and being farther away it looked (A) larger (B) smaller'
context = 'qrel(distance, higher, Train on a bridge) -> qrel(apparentSize, higher, Train on a bridge) ; qrel(apparentSize, lower, Train on a bridge)'

get_response(question, context)

# output: 'smaller'
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
