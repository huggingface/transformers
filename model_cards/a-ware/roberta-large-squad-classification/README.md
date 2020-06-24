---
datasets:
- squad_v2
---

# Roberta-LARGE finetuned on SQuADv2

This is roberta-large model finetuned on SQuADv2 dataset for question answering answerability classification

## Model details
This model is simply an Sequenceclassification model with two inputs (context and question) in a list.
The result is either [1] for answerable or [0] if it is not answerable.
It was trained over 4 epochs on squadv2 dataset and can be used to filter out which context is good to give into the QA model to avoid bad answers.

## Model training
This model was trained with following parameters using simpletransformers wrapper:
```
train_args = {
    'learning_rate': 1e-5,
    'max_seq_length': 512,
    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    'train_batch_size': 4,
    'num_train_epochs': 4,
    'gradient_accumulation_steps': 2,
    'no_cache': True,
    'use_cached_eval_features': False,
    'save_model_every_epoch': False,
    'output_dir': "bart-squadv2",
    'eval_batch_size': 8,
    'fp16_opt_level': 'O2',
    }
```

## Results
```{"accuracy": 90.48%}```
## Model in Action  🚀
```python3
from simpletransformers.classification import ClassificationModel

model = ClassificationModel('roberta', 'a-ware/roberta-large-squadv2', num_labels=2, args=train_args)

predictions, raw_outputs = model.predict([["my dog is an year old. he loves to go into the rain", "how old is my dog ?"]])
print(predictions)
==> [1]
```

> Created with ❤️ by A-ware UG [![Github icon](https://cdn0.iconfinder.com/data/icons/octicons/1024/mark-github-32.png)](https://github.com/aware-ai)
