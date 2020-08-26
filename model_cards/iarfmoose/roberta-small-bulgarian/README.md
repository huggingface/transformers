---
language: bg
---

# RoBERTa-small-bulgarian


The RoBERTa model was originally introduced in [this paper](https://arxiv.org/abs/1907.11692). This is a smaller version of [RoBERTa-base-bulgarian](https://huggingface.co/iarfmoose/roberta-small-bulgarian) with only 6 hidden layers, but similar performance.

## Intended uses

This model can be used for cloze tasks (masked language modeling) or finetuned on other tasks in Bulgarian.

## Limitations and bias

The training data is unfiltered text from the internet and may contain all sorts of biases.

## Training data

This model was trained on the following data:
- [bg_dedup from OSCAR](https://oscar-corpus.com/)
- [Newscrawl 1 million sentences 2017 from Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/bulgarian)
- [Wikipedia 1 million sentences 2016 from Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/bulgarian)

## Training procedure

The model was pretrained using a masked language-modeling objective with dynamic masking as described [here](https://huggingface.co/roberta-base#preprocessing)

It was trained for 160k steps. The batch size was limited to 8 due to GPU memory limitations.
