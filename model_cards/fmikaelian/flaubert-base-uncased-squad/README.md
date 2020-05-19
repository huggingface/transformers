---
language: french
---

# flaubert-base-uncased-squad

## Description

A baseline model for question-answering in french ([flaubert](https://github.com/getalp/Flaubert) model fine-tuned on [french-translated SQuAD 1.1 dataset](https://github.com/Alikabbadj/French-SQuAD))

## Training hyperparameters

```shell
python3 ./examples/question-answering/run_squad.py \
--model_type flaubert \
--model_name_or_path flaubert-base-uncased \
--do_train \
--do_eval \
--do_lower_case \
--train_file SQuAD-v1.1-train_fr_ss999_awstart2_net.json \
--predict_file SQuAD-v1.1-dev_fr_ss999_awstart2_net.json \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir output \
--per_gpu_eval_batch_size=3 \
--per_gpu_train_batch_size=3
``` 

## Evaluation results

```shell
{"f1": 68.66174806561969, "exact_match": 49.299692063176714}
```

## Usage

```python
from transformers import pipeline

nlp = pipeline('question-answering', model='fmikaelian/flaubert-base-uncased-squad', tokenizer='fmikaelian/flaubert-base-uncased-squad')

nlp({
    'question': "Qui est Claude Monet?",
    'context': "Claude Monet, né le 14 novembre 1840 à Paris et mort le 5 décembre 1926 à Giverny, est un peintre français et l’un des fondateurs de l'impressionnisme."
})
```