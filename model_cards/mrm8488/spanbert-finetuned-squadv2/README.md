---
language: english
thumbnail:
---

# SpanBERT (spanbert-base-cased) fine-tuned on SQuAD v2

[SpanBERT](https://github.com/facebookresearch/SpanBERT) created by [Facebook Research](https://github.com/facebookresearch) and fine-tuned on [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) for **Q&A** downstream task.

## Details of SpanBERT

[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)

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
| **EM** | **78.80** |
| **F1** | **82.22** |

### Raw metrics:

```json
{
  "exact": 78.80064010780762,
  "f1": 82.22801347271162,
  "total": 11873,
  "HasAns_exact": 78.74493927125506,
  "HasAns_f1": 85.60951483831069,
  "HasAns_total": 5928,
  "NoAns_exact": 78.85618166526493,
  "NoAns_f1": 78.85618166526493,
  "NoAns_total": 5945,
  "best_exact": 78.80064010780762,
  "best_exact_thresh": 0.0,
  "best_f1": 82.2280134727116,
  "best_f1_thresh": 0.0
}
```

## Comparison:

| Model                                                                                     | EM        | F1 score  |
| ----------------------------------------------------------------------------------------- | --------- | --------- |
| [SpanBert official repo](https://github.com/facebookresearch/SpanBERT#pre-trained-models) | -         | 83.6\*    |
| [spanbert-finetuned-squadv2](https://huggingface.co/mrm8488/spanbert-finetuned-squadv2)   | **78.80** | **82.22** |

## Model in action

Fast usage with **pipelines**:

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/spanbert-finetuned-squadv2",
    tokenizer="mrm8488/spanbert-finetuned-squadv2"
)

qa_pipeline({
    'context': "Manuel Romero has been working hardly in the repository hugginface/transformers lately",
    'question': "Who has been working hard for hugginface/transformers lately?"

})

# Output: {'answer': 'Manuel Romero','end': 13,'score': 6.836378586818937e-09, 'start': 0}
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
