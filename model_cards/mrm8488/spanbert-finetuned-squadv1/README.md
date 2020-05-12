---
language: english
thumbnail:
---

# SpanBERT (spanbert-base-cased) fine-tuned on SQuAD v1.1


[SpanBERT](https://github.com/facebookresearch/SpanBERT) created by [Facebook Research](https://github.com/facebookresearch) and fine-tuned on [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/) for **Q&A** downstream task.

## Details of SpanBERT

 A pre-training method that is designed to better represent and predict spans of text.

[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)

## Details of the downstream task (Q&A) - Dataset

[SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/) contains 100,000+ question-answer pairs on 500+ articles.

| Dataset  | Split | # samples |
| -------- | ----- | --------- |
| SQuAD1.1 | train | 87.7k     |
| SQuAD1.1 | eval  | 10.6k     |

## Model training

The model was trained on a Tesla P100 GPU and 25GB of RAM.
The script for fine tuning can be found [here](https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_squad.py)

## Results:

| Metric | # Value   |
| ------ | --------- |
| **EM** | **85.49** |
| **F1** | **91.98** |

### Raw metrics:

```json
{
  "exact": 85.49668874172185,
  "f1": 91.9845699540379,
  "total": 10570,
  "HasAns_exact": 85.49668874172185,
  "HasAns_f1": 91.9845699540379,
  "HasAns_total": 10570,
  "best_exact": 85.49668874172185,
  "best_exact_thresh": 0.0,
  "best_f1": 91.9845699540379,
  "best_f1_thresh": 0.0
}
```

## Comparison:

| Model                                                                                     | EM        | F1 score  |
| ----------------------------------------------------------------------------------------- | --------- | --------- |
| [SpanBert official repo](https://github.com/facebookresearch/SpanBERT#pre-trained-models) | -         | 92.4\* |
| [spanbert-finetuned-squadv1](https://huggingface.co/mrm8488/spanbert-finetuned-squadv1)   | **85.49** | **91.98** |

## Model in action

Fast usage with **pipelines**:

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/spanbert-finetuned-squadv1",
    tokenizer="mrm8488/spanbert-finetuned-squadv1"
)

qa_pipeline({
    'context': "Manuel Romero has been working hardly in the repository hugginface/transformers lately",
    'question': "Who has been working hard for hugginface/transformers lately?"

})
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/) 

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
