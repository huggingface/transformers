# TF-Adapter-BERT
Learn more in the paper ["Parameter-Efficient Transfer Learning for NLP"](https://arxiv.org/abs/1902.00751).


## Usage
An example of training adapters in BERT's encoders for MRPC classification task:
```
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

## Results on GLUE test set
| CoLA  | SST-2  | MRPC  | STS-B  | QQP  | MNLI(m)  | MNLI(mm)  | QNLI  | RTE  | Total |
| ----|:----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:|
| 55.2 | 90.9 | 88.7 | 82.5 | 71.4 | 84.1 | 83.3 | 90.6 | 68.2 |79.4 |
