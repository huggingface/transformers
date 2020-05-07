---
language: english
thumbnail:
---

# BERT-Small fine-tuned on SQuAD v2

[BERT-Small](https://github.com/google-research/bert/) created by [Google Research](https://github.com/google-research) and fine-tuned on [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) for **Q&A** downstream task.

**Mode size** (after training): **109.74 MB**

## Details of BERT-Small and its 'family' (from their documentation)

Released on March 11th, 2020

This is model is a part of 24 smaller BERT models (English only, uncased, trained with WordPiece masking) referenced in [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962).

The smaller BERT models are intended for environments with restricted computational resources. They can be fine-tuned in the same manner as the original BERT models. However, they are most effective in the context of knowledge distillation, where the fine-tuning labels are produced by a larger and more accurate teacher.

## Details of the downstream task (Q&A) - Dataset

[SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

| Dataset  | Split | # samples |
| -------- | ----- | --------- |
| SQuAD2.0 | train | 130k      |
| SQuAD2.0 | eval  | 12.3k     |

## Model training

The model was trained on a Tesla P100 GPU and 25GB of RAM.
The script for fine tuning can be found [here](https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_squad.py)

## Results:

| Metric | # Value   |
| ------ | --------- |
| **EM** | **60.49** |
| **F1** | **64.21** |

## Comparison:

| Model                                                                                       | EM        | F1 score  | SIZE (MB) |
| ------------------------------------------------------------------------------------------- | --------- | --------- | --------- |
| [bert-tiny-finetuned-squadv2](https://huggingface.co/mrm8488/bert-tiny-finetuned-squadv2)   | 48.60     | 49.73     | **16.74** |
| [bert-mini-finetuned-squadv2](https://huggingface.co/mrm8488/bert-mini-finetuned-squadv2)   | 56.31     | 59.65     | 42.63     |
| [bert-small-finetuned-squadv2](https://huggingface.co/mrm8488/bert-small-finetuned-squadv2) | **60.49** | **64.21** | 109.74    |

## Model in action

Fast usage with **pipelines**:

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-small-finetuned-squadv2",
    tokenizer="mrm8488/bert-small-finetuned-squadv2"
)

qa_pipeline({
    'context': "Manuel Romero has been working hardly in the repository hugginface/transformers lately",
    'question': "Who has been working hard for hugginface/transformers lately?"

})

# Output:
```

```json
{
  "answer": "Manuel Romero",
  "end": 13,
  "score": 0.9939319924374637,
  "start": 0
}
```

### Yes! That was easy 🎉 Let's try with another example

```python
qa_pipeline({
    'context': "Manuel Romero has been working hardly in the repository hugginface/transformers lately",
    'question': "For which company has worked Manuel Romero?"
})

# Output:
```

```json
{
  "answer": "hugginface/transformers",
  "end": 79,
  "score": 0.6024888734447131,
  "start": 56
}
```

### It works!! 🎉 🎉 🎉

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
