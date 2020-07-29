---
language: en
tags:
- bart
- text-classification
- pytorch
datasets:
- yahoo-answers
---

# bart-lage-mnli-yahoo-answers

## Model Description

This model takes [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) and fine-tunes it on Yahoo Answers topic classification. It can be used to predict whether a topic label can be assigned to a given sequence, whether or not the label has been seen before.

It is fine-tuned on Yahoo Answers in the manner originally described in [Yin et al. 2019](https://arxiv.org/abs/1909.00161) and [this blog post](https://joeddav.github.io/blog/2020/05/29/ZSL.html). That is, each sequence is fed to the pre-trained NLI model in place of the premise and each candidate label as the hypothesis, formatted like so: `This text is about {class name}.` For each example in the training set, a true and a randomly-selected false label hypothesis are fed to the model which must predict which labels are valid and which are false.

You can play with an interactive demo of this zero-shot technique with this model, as well as the non-finetuned [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli), [here](https://huggingface.co/zero-shot/).

Since this method studies the ability to classify unseen labels after being trained on a different set of labels, the model is only trained on 5 out of the 10 labels in Yahoo Answers. These are "Society & Culture", "Health", "Computers & Internet", "Business & Finance", and "Family & Relationships".

## Model Usage

The model can be used with the `zero-shot-classification` pipeline like so:

```python
from transformers import pipeline
nlp = pipeline("zero-shot-classification", model="joeddav/bart-large-mnli-yahoo-answers")

sequence_to_classify = "Who are you voting for in 2020?"
candidate_labels = ["Europe", "public health", "politics", "elections"]
hypothesis_template = "This text is about {}."
nlp(sequence_to_classify, candidate_labels, multi_class=True, hypothesis_template=hypothesis_template)
```
