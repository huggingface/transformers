# BERT-base-cased-qa-evaluator

This model takes a question answer pair as an input and outputs a value representing its prediction about whether the input was a valid question and answer pair or not. The model is a pretrained [BERT-base-cased](https://huggingface.co/bert-base-cased) with a sequence classification head.

## Intended uses

The QA evaluator was originally designed to be used with the [t5-base-question-generator](https://huggingface.co/iarfmoose/t5-base-question-generator) for evaluating the quality of generated questions. 

The input for the QA evaluator follows the format for `BertForSequenceClassification`, but using the question and answer as the two sequences. Inputs should take the following format:
```
[CLS] <question> [SEP] <answer [SEP]
```

## Limitations and bias

The model is trained to evaluate if a question and answer are semantically related, but cannot determine whether an answer is actually true/correct or not.

## Training data

The training data was made up of question-answer pairs from the following datasets: 
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- [RACE](http://www.cs.cmu.edu/~glai1/data/race/)
- [CoQA](https://stanfordnlp.github.io/coqa/)
- [MSMARCO](https://microsoft.github.io/msmarco/)

## Training procedure

The question and answer were concatenated 50% of the time. In the other 50% of the time a corruption operation was performed (either swapping the answer for an unrelated answer, or by copying part of the question into the answer). The model was then trained to predict whether the input sequence represented one of the original QA pairs or a corrupted input.
