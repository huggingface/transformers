## RoBERTa-large language model fine-tuned on SQuAD2.0

### with the following results:

```
  "exact": 84.02257222269014,
  "f1": 87.47063479332766,
  "total": 11873,
  "HasAns_exact": 81.19095816464238,
  "HasAns_f1": 88.0969714745582,
  "HasAns_total": 5928,
  "NoAns_exact": 86.84608915054667,
  "NoAns_f1": 86.84608915054667,
  "NoAns_total": 5945,
  "best_exact": 84.02257222269014,
  "best_exact_thresh": 0.0,
  "best_f1": 87.47063479332759,
  "best_f1_thresh": 0.0
```
### from script:
```
python -m torch.distributed.launch --nproc_per_node=2 ${RUN_SQUAD_DIR}/run_squad.py \
  --model_type roberta \
  --model_name_or_path roberta-large \
  --do_train \
  --train_file ${SQUAD_DIR}/train-v2.0.json \
  --predict_file ${SQUAD_DIR}/dev-v2.0.json \
  --version_2_with_negative \
  --num_train_epochs 2 \
  --warmup_steps 328 \
  --weight_decay 0.01 \
  --do_lower_case \
  --learning_rate 1.5e-5 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --save_steps 1000 \
  --per_gpu_train_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --logging_steps 50 \
  --threads 10 \
  --overwrite_cache \
  --overwrite_output_dir \
  --output_dir ${MODEL_PATH}

python ${RUN_SQUAD_DIR}/run_squad.py \
  --model_type roberta \
  --model_name_or_path ${MODEL_PATH} \
  --do_eval \
  --train_file ${SQUAD_DIR}/train-v2.0.json \
  --predict_file ${SQUAD_DIR}/dev-v2.0.json \
  --version_2_with_negative \
  --do_lower_case \
  --max_seq_length 512 \
  --per_gpu_eval_batch_size 24 \
  --eval_all_checkpoints \
  --overwrite_output_dir \
  --output_dir ${MODEL_PATH}
$@
```
### using the following system & software:
```
OS/Platform: Linux-4.15.0-91-generic-x86_64-with-debian-buster-sid
GPU/CPU: 2 x NVIDIA 1080Ti / Intel i7-8700
Transformers: 2.7.0
PyTorch: 1.4.0
TensorFlow: 2.1.0
Python: 3.7.7
```
