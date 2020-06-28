---
datasets:
- squad_v2
---

# XLM-ROBERTA-LARGE finetuned on SQuADv2

This is xlm-roberta-large model finetuned on SQuADv2 dataset for question answering task

## Model details
XLM-Roberta was propsed in the [paper](https://arxiv.org/pdf/1911.02116.pdf) **XLM-R: State-of-the-art cross-lingual understanding through self-supervision

## Model training
This model was trained with following parameters using simpletransformers wrapper:
```
train_args = {
    'learning_rate': 1e-5,
    'max_seq_length': 512,
    'doc_stride': 512,
    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    'train_batch_size': 8,
    'num_train_epochs': 2,
    'gradient_accumulation_steps': 2,
    'no_cache': True,
    'use_cached_eval_features': False,
    'save_model_every_epoch': False,
    'output_dir': "bart-squadv2",
    'eval_batch_size': 32,
    'fp16_opt_level': 'O2',
    }
```

## Results
```{"correct": 6961, "similar": 4359, "incorrect": 553, "eval_loss": -12.177856394381962}```

## Model in Action  🚀
```python3
from transformers import XLMRobertaTokenizer, XLMRobertaForQuestionAnswering
import torch

tokenizer = XLMRobertaTokenizer.from_pretrained('a-ware/xlmroberta-squadv2')
model = XLMRobertaForQuestionAnswering.from_pretrained('a-ware/xlmroberta-squadv2')

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
encoding = tokenizer(question, text, return_tensors='pt')
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

start_scores, end_scores = model(input_ids, attention_mask=attention_mask, output_attentions=False)[:2]

all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
answer = tokenizer.convert_tokens_to_ids(answer.split())
answer = tokenizer.decode(answer)
#answer => 'a nice puppet' 
```

> Created with ❤️ by A-ware UG [![Github icon](https://cdn0.iconfinder.com/data/icons/octicons/1024/mark-github-32.png)](https://github.com/aware-ai)
