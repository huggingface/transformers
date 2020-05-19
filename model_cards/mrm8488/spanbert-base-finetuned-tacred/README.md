---
language: english
thumbnail:
---

# SpanBERT base fine-tuned on TACRED

[SpanBERT](https://github.com/facebookresearch/SpanBERT) created by [Facebook Research](https://github.com/facebookresearch) and fine-tuned on [TACRED](https://nlp.stanford.edu/projects/tacred/) dataset by [them](https://github.com/facebookresearch/SpanBERT#finetuned-models-squad-1120-relation-extraction-coreference-resolution)

## Details of SpanBERT

[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)

## Dataset 📚

[TACRED](https://nlp.stanford.edu/projects/tacred/) A large-scale relation extraction dataset with 106k+ examples over 42 TAC KBP relation types.

## Model fine-tuning 🏋️‍

You can get the fine-tuning script [here](https://github.com/facebookresearch/SpanBERT)

```bash
python code/run_tacred.py \
  --do_train \
  --do_eval \
  --data_dir <TACRED_DATA_DIR> \
  --model spanbert-base-cased \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --max_seq_length 128 \
  --output_dir tacred_dir \
  --fp16
```

## Results Comparison 📝

|                   | SQuAD 1.1     | SQuAD 2.0  | Coref   | TACRED |
| ----------------------  | ------------- | ---------  | ------- | ------ |
|                         | F1            | F1         | avg. F1 |  F1    |
| BERT (base)             | 88.5*         | 76.5*      | 73.1    |  67.7  |
| SpanBERT (base)         | [92.4*](https://huggingface.co/mrm8488/spanbert-base-finetuned-squadv1)         | [83.6*](https://huggingface.co/mrm8488/spanbert-base-finetuned-squadv2)      | 77.4    |  **68.2** (this one)  |
| BERT (large)            | 91.3          | 83.3       | 77.1    |  66.4  |
| SpanBERT (large)        | [94.6](https://huggingface.co/mrm8488/spanbert-large-finetuned-squadv1)        | [88.7](https://huggingface.co/mrm8488/spanbert-large-finetuned-squadv2)     | 79.6    |  [70.8](https://huggingface.co/mrm8488/spanbert-base-finetuned-tacred)   |


Note: The numbers marked as * are evaluated on the development sets becaus those models were not submitted to the official SQuAD leaderboard. All the other numbers are test numbers.


> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
