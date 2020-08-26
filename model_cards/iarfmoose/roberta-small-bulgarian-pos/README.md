---
language: bg
---

# RoBERTa-small-bulgarian-POS


The RoBERTa model was originally introduced in [this paper](https://arxiv.org/abs/1907.11692). This model is a version of [RoBERTa-small-Bulgarian](https://huggingface.co/iarfmoose/roberta-small-bulgarian) fine-tuned for part-of-speech tagging.

## Intended uses

The model can be used to predict part-of-speech tags in Bulgarian text. Since the tokenizer uses byte-pair encoding, each word in the text may be split into more than one token. When predicting POS-tags, the last token from each word can be used. Using the last token was found to slightly outperform predictions based on the first token.

An example of this can be found [here](https://github.com/iarfmoose/bulgarian-nlp/blob/master/models/postagger.py).

## Limitations and bias

The pretraining data is unfiltered text from the internet and may contain all sorts of biases.

## Training data

In addition to the pretraining data used in [RoBERTa-base-Bulgarian]([RoBERTa-base-Bulgarian](https://huggingface.co/iarfmoose/roberta-base-bulgarian)), the model was trained on the UPOS tags from (UD_Bulgarian-BTB)[https://github.com/UniversalDependencies/UD_Bulgarian-BTB].

## Training procedure

The model was trained for 5 epochs over the training set. The loss was calculated based on label predictions for the last POS-tag for each word. The model achieves 98% on the test set.
