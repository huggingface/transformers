# albert-chinese-large-qa
Albert large QA model pretrained from baidu webqa and baidu dureader datasets.

## Data source
+ baidu webqa 1.0
+ baidu dureader

## Traing Method
We combined the two datasets together and created a new dataset in squad format, including 705139 samples for training and 69638 samples for validation.
We finetune the model based on the albert chinese large model.

## Hyperparams
+ learning_rate 1e-5
+ max_seq_length 512
+ max_query_length 50
+ max_answer_length 300
+ doc_stride 256
+ num_train_epochs 2
+ warmup_steps 1000
+ per_gpu_train_batch_size 8
+ gradient_accumulation_steps 3
+ n_gpu 2 (Nvidia Tesla P100)

## Usage
```
from transformers import AutoModelForQuestionAnswering, BertTokenizer

model = AutoModelForQuestionAnswering.from_pretrained('wptoux/albert-chinese-large-qa')
tokenizer = BertTokenizer.from_pretrained('wptoux/albert-chinese-large-qa')
```
***Important: use BertTokenizer***

## MoreInfo
Please visit https://github.com/wptoux/albert-chinese-large-webqa for details.
