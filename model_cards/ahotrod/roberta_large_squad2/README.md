## RoBERTa-large language model fine-tuned on SQuAD2.0

### with the following results:

```
  "exact": 84.46896319380106,
  "f1": 87.85388093408943,
  "total": 11873,
  "HasAns_exact": 81.37651821862349,
  "HasAns_f1": 88.1560607844881,
  "HasAns_total": 5928,
  "NoAns_exact": 87.55256518082422,
  "NoAns_f1": 87.55256518082422,
  "NoAns_total": 5945,
  "best_exact": 84.46896319380106,
  "best_exact_thresh": 0.0,
  "best_f1": 87.85388093408929,
  "best_f1_thresh": 0.0
```
### from script:
```
python ${EXAMPLES}/run_squad.py \
  --model_type roberta \
  --model_name_or_path roberta-large \
  --do_train \
  --do_eval \
  --train_file ${SQUAD}/train-v2.0.json \
  --predict_file ${SQUAD}/dev-v2.0.json \
  --version_2_with_negative \
  --do_lower_case \
  --num_train_epochs 3 \
  --warmup_steps 1642 \
  --weight_decay 0.01 \
  --learning_rate 3e-5 \
  --adam_epsilon 1e-6 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_gpu_train_batch_size 8 \
  --gradient_accumulation_steps 6 \
  --per_gpu_eval_batch_size 48 \
  --threads 12 \
  --logging_steps 50 \
  --save_steps 2000 \
  --overwrite_output_dir \
  --output_dir ${MODEL_PATH}
$@
```
### using the following system & software:
```
Transformers: 2.7.0
PyTorch: 1.4.0
TensorFlow: 2.1.0
Python: 3.7.7
OS/Platform: Linux-5.3.0-46-generic-x86_64-with-debian-buster-sid
CPU/GPU: Intel i9-9900K / NVIDIA Titan RTX 24GB
```
