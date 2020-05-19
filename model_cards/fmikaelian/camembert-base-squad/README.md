---
language: french
---

# camembert-base-squad

## Description

A baseline model for question-answering in french ([CamemBERT](https://camembert-model.fr/) model fine-tuned on [french-translated SQuAD 1.1 dataset](https://github.com/Alikabbadj/French-SQuAD))

## Training hyperparameters

```shell
python3 ./examples/question-answering/run_squad.py \
--model_type camembert \
--model_name_or_path camembert-base \
--do_train \
--do_eval \
--do_lower_case \
--train_file SQuAD-v1.1-train_fr_ss999_awstart2_net.json \
--predict_file SQuAD-v1.1-dev_fr_ss999_awstart2_net.json \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir output3 \
--per_gpu_eval_batch_size=3 \
--per_gpu_train_batch_size=3 \
--save_steps 10000
``` 

## Evaluation results

```shell
{"f1": 79.8570684959745, "exact_match": 59.21327108373895}
```

## Usage

```python
from transformers import pipeline

nlp = pipeline('question-answering', model='fmikaelian/camembert-base-squad', tokenizer='fmikaelian/camembert-base-squad')

nlp({
    'question': "Qui est Claude Monet?",
    'context': "Claude Monet, né le 14 novembre 1840 à Paris et mort le 5 décembre 1926 à Giverny, est un peintre français et l’un des fondateurs de l'impressionnisme."
})
```