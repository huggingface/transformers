---
datasets:
- squad_v2
---

# BART-LARGE finetuned on SQuADv2

This is bart-large model finetuned on SQuADv2 dataset for question answering task

## Model details
BART was propsed in the [paper](https://arxiv.org/abs/1910.13461) **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**.
BART is a seq2seq model intended for both NLG and NLU tasks. 

To use BART for question answering tasks, we feed the complete document into the encoder and decoder, and use the top
hidden state of the decoder as a representation for each
word. This representation is used to classify the token. As given in the paper bart-large achives comparable to ROBERTa on SQuAD.
Another notable thing about BART is that it can handle sequences with upto 1024 tokens.

| Param               | #Value |
|---------------------|--------|
| encoder layers      | 12     |
| decoder layers      | 12     |
| hidden size         | 4096   |
| num attetion heads  | 16     |
| on disk size        | 1.63GB |


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

[You can even train your own model using this colab notebook](https://colab.research.google.com/drive/1I5cK1M_0dLaf5xoewh6swcm5nAInfwHy?usp=sharing)

## Results
```{"correct": 6832, "similar": 4409, "incorrect": 632, "eval_loss": -14.950117511952177}```

## Model in Action  🚀
```python3
from transformers import BartTokenizer, BartForQuestionAnswering
import torch

tokenizer = BartTokenizer.from_pretrained('a-ware/bart-squadv2')
model = BartForQuestionAnswering.from_pretrained('a-ware/bart-squadv2')

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
