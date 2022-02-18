<!---
Copyright 2022 The Microsoft Inc. and The HuggingFace Inc. Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Run Table Tasks with TAPEX

TAPEX is a table pre-training approach for table-related tasks. By learning a neural SQL executor over a synthetic corpus based on generative language models (e.g., BART), it achieves state-of-the-art performance on several table-based question answering benchmarks and table-based fact verification benchmark. More details can be found in the original paper [TAPEX: Table Pre-training via Learning a Neural SQL Executor](https://arxiv.org/pdf/2107.07653.pdf).

> If you are also familiar with fairseq, you may also find useful [the official implementation in fairseq](https://github.com/microsoft/Table-Pretraining).

## Table Question Answering Tasks

### What is Table Question Answering

![Example](https://table-pretraining.github.io/assets/tableqa_task.png)

The task of Table Question Answering (TableQA) is to empower machines to answer users' questions over a given table. The resulting answer(s) can be a region in the table, or a number calculated by applying aggregation operators to a specific region.

### What Questions Can be Answered

Benefiting from the powerfulness of generative models, TAPEX can deal with almost all kinds of questions over tables (if there is training data). Below are some typical question and their answers taken from [WikiTableQuestion](https://nlp.stanford.edu/blog/wikitablequestions-a-complex-real-world-question-understanding-dataset).

| Question | Answer |
| :---: | :---: |
| What is the years won for each team? | 2004, 2008, 2012 |
| How long did Taiki Tsuchiya last? | 4:27 |
| What is the total amount of matches drawn? | 1 |
| Besides Tiger Woods, what other player won between 2007 and 2009? | Camilo Villegas |
| What was the last Baekje Temple? | Uija |
| What is the difference between White voters and Black voters in 1948? | 0 |
| What is the average number of sailors for each country during the worlds qualification tournament? | 2 |


### How to Fine-tune TAPEX on TableQA

We provide a fine-tuning script of tapex for TableQA on the WikiSQL benchmark: [WikiSQL](https://github.com/salesforce/WikiSQL).
This script is customized for tapex models, and can be easily adapted to other benchmarks such as WikiTableQuestion
(only some tweaks in the function `preprocess_tableqa_function`).

Here is how to run the script on the WikiSQL:

```bash
export EXP_NAME=wikisql_tapex_base

python run_wikisql_with_tapex.py \
  --do_train \
  --do_eval \
  --output_dir $EXP_NAME \
  --model_name_or_path microsoft/tapex-base \
  --overwrite_output_dir \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 6 \
  --num_train_epochs 100 \
  --warmup_ratio 0.1 \
  --logging_steps 10 \
  --learning_rate 3e-5 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --max_steps 10000
```

## Table Fact Verification Tasks

### What is Table Fact Verification

![Example](https://table-pretraining.github.io/assets/tableft_task.png)

The task of Table Fact Verification (TableFV) is to empower machines to justify if a statement follows facts in a given table. The result is a binary classification belonging to `1` (entailed) or `0` (refused).

### How to Fine-tune TAPEX on TableFV

We provide a fine-tuning script of tapex for TableFV on the TabFact benchmark: [TabFact](https://github.com/wenhuchen/Table-Fact-Checking).

Here is how to run the script on the TabFact:

```bash
export EXP_NAME=tabfact_tapex_base

python run_tabfact_with_tapex.py \
  --do_train \
  --do_eval \
  --output_dir $EXP_NAME \
  --model_name_or_path microsoft/tapex-base \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 6 \
  --gradient_accumulation_steps 12 \
  --num_train_epochs 15 \
  --warmup_ratio 0.1 \
  --logging_steps 10 \
  --learning_rate 3e-5 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --evaluation_strategy steps \
  --max_steps 20000
```
