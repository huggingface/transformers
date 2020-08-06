# Adapters
## What is Adapter?

[Houlsby et al. (2019)](https://arxiv.org/abs/1902.00751) introduced adapters as an alternative approach for adaptation in transfer learning in NLP within deep transformer-based architectures.
Adapters are task-specific neural modules that are added between layers of a pre-trained network. After coping weights from a pre-trained network, pre-trained weights will be frozen, and only Adapters will be trained.

## Why Adapter?
Adapters provide numerous benefits over plain fully fine-tuning or other approaches that result in compact models such as multi-task learning:

* It is a lightweight alternative to fully fine-tuning that trains only a few trainable parameters per task without sacrificing performance.
* Yielding a high degree of parameter sharing between down-stream tasks due to being frozen of original network parameters.
* Unlike multi-task learning that requires simultaneous access to all tasks, it allows training on down-stream tasks sequentially. Thus, adding new tasks do not require complete joint retraining. Further, eliminates the hassle of weighing losses or balancing training set sizes.
* Training adapters for each task separately, leading to that the model not forgetting how to perform previous tasks (the problem of catastrophic forgetting).


Learn more in the paper ["Parameter-Efficient Transfer Learning for NLP"](https://arxiv.org/abs/1902.00751).


## Training
An example of training adapters in BERT's encoders for MRPC classification task:
```bash
python run_tf_adapter_bert.py \
  --casing bert-base-uncased \
  --bottleneck_size 32\
  --non_linearity gelu\
  --task mrpc \
  --batch_size 32 \
  --epochs 10 \
  --max_seq_length 128 \
  --learning_rate 3e-4 \
  --warmup_ratio 0.1 \
  --saved_models_dir "saved_models"\
```

## Results
 on GLUE test set:
| CoLA  | SST\-2  | MRPC  | STS\-B  | QQP  | MNLI(m)  | MNLI(mm)  | QNLI  | RTE  | Total |
| ----|:----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:|
| 55\.2 | 90\.9 | 88\.7 | 82\.5 | 71\.4 | 84\.1 | 83\.3 | 90\.6 | 68\.2 |79\.4 |
