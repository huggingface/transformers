---
language: english
thumbnail:
---

# SpanBERT base fine-tuned on SQuAD v2

[SpanBERT](https://github.com/facebookresearch/SpanBERT) created by [Facebook Research](https://github.com/facebookresearch) and fine-tuned on [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) for **Q&A** downstream task ([by them](https://github.com/facebookresearch/SpanBERT#finetuned-models-squad-1120-relation-extraction-coreference-resolution)).

## Details of SpanBERT

[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)

## Details of the downstream task (Q&A) - Dataset 📚 🧐 ❓

[SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

| Dataset  | Split | # samples |
| -------- | ----- | --------- |
| SQuAD2.0 | train | 130k      |
| SQuAD2.0 | eval  | 12.3k     |

## Model fine-tuning 🏋️‍

You can get the fine-tuning script [here](https://github.com/facebookresearch/SpanBERT)

```bash
python code/run_squad.py \
  --do_train \
  --do_eval \
  --model spanbert-base-cased \
  --train_file train-v2.0.json \
  --dev_file dev-v2.0.json \
  --train_batch_size 32 \
  --eval_batch_size 32  \
  --learning_rate 2e-5 \
  --num_train_epochs 4 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --eval_metric best_f1 \
  --output_dir squad2_output \
  --version_2_with_negative \
  --fp16
```

## Results Comparison 📝

|                   | SQuAD 1.1     | SQuAD 2.0  | Coref   | TACRED |
| ----------------------  | ------------- | ---------  | ------- | ------ |
|                         | F1            | F1         | avg. F1 |  F1    |
| BERT (base)             | 88.5         | 76.5      | 73.1    |  67.7  |
| SpanBERT (base)         | [92.4](https://huggingface.co/mrm8488/spanbert-base-finetuned-squadv1)         | **83.6** (this one)      | 77.4    |  [68.2](https://huggingface.co/mrm8488/spanbert-base-finetuned-tacred)  |
| BERT (large)            | 91.3          | 83.3       | 77.1    |  66.4  |
| SpanBERT (large)        | [94.6](https://huggingface.co/mrm8488/spanbert-large-finetuned-squadv1)          | [88.7](https://huggingface.co/mrm8488/spanbert-large-finetuned-squadv2)     | 79.6    |  [70.8](https://huggingface.co/mrm8488/spanbert-large-finetuned-tacred)  |


Note: The numbers marked as * are evaluated on the development sets becaus those models were not submitted to the official SQuAD leaderboard. All the other numbers are test numbers.

## Model in action

Fast usage with **pipelines**:

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/spanbert-base-finetuned-squadv2",
    tokenizer="SpanBERT/spanbert-base-cased"
)

qa_pipeline({
    'context': "Manuel Romero has been working very hard in the repository hugginface/transformers lately",
    'question': "How has been working Manuel Romero lately?"

})
# Output: {'answer': 'very hard', 'end': 40, 'score': 0.9052708846768347, 'start': 31}
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
