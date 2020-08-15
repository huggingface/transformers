---
language: en
datasets:
- yelp_polarity
---

# RoBERTa-base-finetuned-yelp-polarity

This is a [RoBERTa-base](https://huggingface.co/roberta-base) checkpoint fine-tuned on binary sentiment classifcation from [Yelp polarity](https://huggingface.co/nlp/viewer/?dataset=yelp_polarity).
It gets **98.08%** accuracy on the test set.

## Hyper-parameters

We used the following hyper-parameters to train the model on one GPU:
```python
num_train_epochs            = 2.0
learning_rate               = 1e-05
weight_decay                = 0.0
adam_epsilon                = 1e-08
max_grad_norm               = 1.0
per_device_train_batch_size = 32
gradient_accumulation_steps = 1
warmup_steps                = 3500
seed                        = 42
```
