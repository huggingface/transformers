---
language: multilingual
thumbnail:
---

# A fine-tuned model on GoldP task from Tydi QA dataset

This model uses [bert-multi-cased-finetuned-xquadv1](https://huggingface.co/mrm8488/bert-multi-cased-finetuned-xquadv1) and fine-tuned on [Tydi QA](https://github.com/google-research-datasets/tydiqa) dataset for Gold Passage task [(GoldP)](https://github.com/google-research-datasets/tydiqa#the-tasks)

## Details of the language model
The base language model [(bert-multi-cased-finetuned-xquadv1)](https://huggingface.co/mrm8488/bert-multi-cased-finetuned-xquadv1) is a fine-tuned version of [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) for the **Q&A** downstream task


## Details of the Tydi QA dataset

TyDi QA contains 200k human-annotated question-answer pairs in 11 Typologically Diverse languages, written without seeing the answer and without the use of translation, and is designed for the **training and evaluation** of automatic question answering systems. This repository provides evaluation code and a baseline system for the dataset. https://ai.google.com/research/tydiqa


## Details of the downstream task (Gold Passage or GoldP aka the secondary task)

Given a passage that is guaranteed to contain the answer, predict the single contiguous span of characters that answers the question. the gold passage task differs from the [primary task](https://github.com/google-research-datasets/tydiqa/blob/master/README.md#the-tasks) in several ways:
*   only the gold answer passage is provided rather than the entire Wikipedia article;
*   unanswerable questions have been discarded, similar to MLQA and XQuAD;
*   we evaluate with the SQuAD 1.1 metrics like XQuAD; and
*   Thai and Japanese are removed since the lack of whitespace breaks some tools.


## Model training

The model was fine-tuned on a Tesla P100 GPU and 25GB of RAM.
The script is the following:

```python
python run_squad.py \
  --model_type bert \
  --model_name_or_path mrm8488/bert-multi-cased-finetuned-xquadv1 \
  --do_train \
  --do_eval \
  --train_file /content/dataset/train.json \
  --predict_file /content/dataset/dev.json \
  --per_gpu_train_batch_size 24 \
  --per_gpu_eval_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /content/model_output \
  --overwrite_output_dir \
  --save_steps 5000 \
  --threads 40
  ```

## Global Results (dev set):

| Metric    | # Value     |
| --------- | ----------- |
| **Exact** | **71.06** |
| **F1**    | **82.16** |

## Specific Results (per language):

| Language    | # Samples     | # Exact | # F1 |
| --------- | ----------- |--------| ------ |
| Arabic    | 1314  | 73.29 | 84.72 |
| Bengali   | 180   | 64.60 | 77.84 |
| English   | 654   | 72.12 |   82.24   |
| Finnish   | 1031  | 70.14 | 80.36 |
| Indonesian| 773   | 77.25 | 86.36 |
| Korean    | 414   | 68.92 | 70.95 |
| Russian   | 1079    | 62.65 | 78.55 |
| Swahili   | 596   | 80.11 | 86.18 |
| Telegu    | 874   | 71.00 | 84.24 |



> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
