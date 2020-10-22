## Albert xxlarge version 1 language model fine-tuned on SQuAD2.0

###  (updated 30Sept2020) with the following results:

```
exact: 86.11134506864315
f1: 89.35371214945009
total': 11873
HasAns_exact': 83.56950067476383
HasAns_f1': 90.06353312254078
HasAns_total': 5928
NoAns_exact': 88.64592094196804
NoAns_f1': 88.64592094196804
NoAns_total': 5945
best_exact': 86.11134506864315
best_exact_thresh': 0.0
best_f1': 89.35371214944985
best_f1_thresh': 0.0
```

### from script:

```
python ${EXAMPLES}/run_squad.py \
  --model_type albert \
  --model_name_or_path albert-xxlarge-v1 \
  --do_train \
  --do_eval \
  --train_file ${SQUAD}/train-v2.0.json \
  --predict_file ${SQUAD}/dev-v2.0.json \
  --version_2_with_negative \
  --do_lower_case \
  --num_train_epochs 3 \
  --max_steps 8144 \
  --warmup_steps 814 \
  --learning_rate 3e-5 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_gpu_train_batch_size 6 \
  --gradient_accumulation_steps 8 \
  --per_gpu_eval_batch_size 48 \
  --fp16 \
  --fp16_opt_level O1 \
  --threads 12 \
  --logging_steps 50 \
  --save_steps 3000 \
  --overwrite_output_dir \
  --output_dir ${MODEL_PATH}
```

### using the following software & system:

```
Transformers: 3.1.0
PyTorch: 1.6.0
TensorFlow: 2.3.1
Python: 3.8.1
OS: Linux-5.4.0-48-generic-x86_64-with-glibc2.10
CPU/GPU: Intel i9-9900K / NVIDIA Titan RTX 24GB
```
