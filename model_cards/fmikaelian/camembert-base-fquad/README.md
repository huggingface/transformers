---
language: french
---

# camembert-base-fquad

## Description

A baseline model for question-answering in french ([CamemBERT](https://camembert-model.fr/) model fine-tuned on [FQuAD](https://fquad.illuin.tech/))

## Training hyperparameters

```shell
python3 ./examples/question-answering/run_squad.py \
--model_type camembert \
--model_name_or_path camembert-base \
--do_train \
--do_eval \
--do_lower_case \
--train_file train.json \
--predict_file valid.json \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir output \
--per_gpu_eval_batch_size=3 \
--per_gpu_train_batch_size=3 \
--save_steps 10000
``` 

## Evaluation results

```shell
{"f1": 77.24515316052342, "exact_match": 52.82308657465496}
```

## Usage

```python
from transformers import pipeline

nlp = pipeline('question-answering', model='fmikaelian/camembert-base-fquad', tokenizer='fmikaelian/camembert-base-fquad')

nlp({
    'question': "Qui est Claude Monet?",
    'context': "Claude Monet, né le 14 novembre 1840 à Paris et mort le 5 décembre 1926 à Giverny, est un peintre français et l’un des fondateurs de l'impressionnisme."
})
```