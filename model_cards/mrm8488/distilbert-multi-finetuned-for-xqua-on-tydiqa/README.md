---
language: multilingual
thumbnail:
---

# DistilBERT multilingual fine-tuned on TydiQA (GoldP task) dataset for multilingual Q&A 😛🌍❓


## Details of the language model

[distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased)


## Details of the Tydi QA dataset

TyDi QA contains 200k human-annotated question-answer pairs in 11 Typologically Diverse languages, written without seeing the answer and without the use of translation, and is designed for the **training and evaluation** of automatic question answering systems. This repository provides evaluation code and a baseline system for the dataset. https://ai.google.com/research/tydiqa


## Details of the downstream task (Gold Passage or GoldP aka the secondary task)

Given a passage that is guaranteed to contain the answer, predict the single contiguous span of characters that answers the question. the gold passage task differs from the [primary task](https://github.com/google-research-datasets/tydiqa/blob/master/README.md#the-tasks) in several ways:
*   only the gold answer passage is provided rather than the entire Wikipedia article;
*   unanswerable questions have been discarded, similar to MLQA and XQuAD;
*   we evaluate with the SQuAD 1.1 metrics like XQuAD; and
*   Thai and Japanese are removed since the lack of whitespace breaks some tools.


## Model training 💪🏋️‍

The model was fine-tuned on a Tesla P100 GPU and 25GB of RAM.
The script is the following:

```python
python transformers/examples/run_squad.py \
  --model_type distilbert \
  --model_name_or_path distilbert-base-multilingual-cased \
  --do_train \
  --do_eval \
  --train_file /path/to/dataset/train.json \
  --predict_file /path/to/dataset/dev.json \
  --per_gpu_train_batch_size 24 \
  --per_gpu_eval_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /content/model_output \
  --overwrite_output_dir \
  --save_steps 1000 \
  --threads 400
  ```

## Global Results (dev set) 📝

| Metric    | # Value     |
| --------- | ----------- |
| **EM**    | **63.85** |
| **F1**    | **75.70** |

## Specific Results (per language) 🌍📝 

| Language    | # Samples     | # EM | # F1 |
| --------- | ----------- |--------| ------ |
| Arabic    | 1314  | 66.66 | 80.02 |
| Bengali   | 180   | 53.09 | 63.50 |
| English   | 654   | 62.42 | 73.12 |
| Finnish   | 1031  | 64.57 | 75.15 |
| Indonesian| 773   | 67.89 | 79.70 |
| Korean    | 414   | 51.29 | 61.73 |
| Russian   | 1079  | 55.42 | 70.08 |
| Swahili   | 596   | 74.51 | 81.15 |
| Telegu    | 874   | 66.21 | 79.85 |


## Similar models

You can also try [bert-multi-cased-finedtuned-xquad-tydiqa-goldp](https://huggingface.co/mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp) that achieves **F1 = 82.16** and **EM = 71.06** (And of course better marks per language).


> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
