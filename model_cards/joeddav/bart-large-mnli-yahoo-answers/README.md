---
language: en
tags:
- text-classification
- pytorch
datasets:
- yahoo-answers
widget:
- text: "Who are you voting for in 2020? <sep> This text is about politics."
---

# bart-lage-mnli-yahoo-answers

## Model Description

This model takes [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) and fine-tunes it on Yahoo Answers topic classification. It can be used to predict whether a topic label can be assigned to a given sequence, whether or not the label has been seen before.

You can play with an interactive demo of this zero-shot technique with this model, as well as the non-finetuned [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli), [here](https://huggingface.co/zero-shot/).

## Inteded Usage

This model was fine-tuned on topic classification and will perform best at zero-shot topic classification. Use `hypothesis_template="This text is about {}."` as this is the template used during fine-tuning.

For settings other than topic classification, you can use any model pre-trained on MNLI such as [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) or [roberta-large-mnli](https://huggingface.co/roberta-large-mnli) with the same code as written below.

#### With the zero-shot classification pipeline

The model can be used with the `zero-shot-classification` pipeline like so:

```python
from transformers import pipeline
nlp = pipeline("zero-shot-classification", model="joeddav/bart-large-mnli-yahoo-answers")

sequence_to_classify = "Who are you voting for in 2020?"
candidate_labels = ["Europe", "public health", "politics", "elections"]
hypothesis_template = "This text is about {}."
nlp(sequence_to_classify, candidate_labels, multi_class=True, hypothesis_template=hypothesis_template)
```

#### With manual PyTorch

```python
# pose sequence as a NLI premise and label as a hypothesis
from transformers import BartForSequenceClassification, BartTokenizer
nli_model = BartForSequenceClassification.from_pretrained('joeddav/bart-large-mnli-yahoo-answers')
tokenizer = BartTokenizer.from_pretrained('joeddav/bart-large-mnli-yahoo-answers')

premise = sequence
hypothesis = f'This text is about {label}.'

# run through model pre-trained on MNLI
x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                        max_length=tokenizer.max_len,
                        truncation_strategy='only_first')
logits = nli_model(x.to(device))[0]

# we throw away "neutral" (dim 1) and take the probability of
# "entailment" (2) as the probability of the label being true 
entail_contradiction_logits = logits[:,[0,2]]
probs = entail_contradiction_logits.softmax(dim=1)
prob_label_is_true = probs[:,1]
```

## Training

The model is a pre-trained MNLI classifier further fine-tuned on Yahoo Answers topic classification in the manner originally described in [Yin et al. 2019](https://arxiv.org/abs/1909.00161) and [this blog post](https://joeddav.github.io/blog/2020/05/29/ZSL.html). That is, each sequence is fed to the pre-trained NLI model in place of the premise and each candidate label as the hypothesis, formatted like so: `This text is about {class name}.` For each example in the training set, a true and a randomly-selected false label hypothesis are fed to the model which must predict which labels are valid and which are false.

Since this method studies the ability to classify unseen labels after being trained on a different set of labels, the model is only trained on 5 out of the 10 labels in Yahoo Answers. These are "Society & Culture", "Health", "Computers & Internet", "Business & Finance", and "Family & Relationships".

## Evaluation Results

This model was evaluated with the label-weighted F1 of the _seen_ and _unseen_ labels. That is, for each example the model must predict from one of the 10 corpus labels. The F1 is reported for the labels seen during training as well as the labels unseen during training. We found an F1 score of `.68` and `.72` for the unseen and seen labels, respectively. In order to adjust for the in-vs-out of distribution labels, we subtract a fixed amount of 30% from the normalized probabilities of the _seen_ labels, as described in [Yin et al. 2019](https://arxiv.org/abs/1909.00161) and [our blog post](https://joeddav.github.io/blog/2020/05/29/ZSL.html).
