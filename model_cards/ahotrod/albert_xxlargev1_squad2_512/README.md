## Albert xxlarge version 1 language model fine-tuned on SQuAD2.0

### with the following results:

```
exact: 85.65653162637918
f1: 89.260458954177
total': 11873
HasAns_exact': 82.6417004048583
HasAns_f1': 89.8598902096736
HasAns_total': 5928
NoAns_exact': 88.66274179983179
NoAns_f1': 88.66274179983179
NoAns_total': 5945
best_exact': 85.65653162637918
best_exact_thresh': 0.0
best_f1': 89.2604589541768
best_f1_thresh': 0.0
```

### from script:

```
python -m torch.distributed.launch --nproc_per_node=2 ${RUN_SQUAD_DIR}/run_squad.py \
--model_type albert \
--model_name_or_path albert-xxlarge-v1 \
--do_train \
--train_file ${SQUAD_DIR}/train-v2.0.json \
--predict_file ${SQUAD_DIR}/dev-v2.0.json \
--version_2_with_negative \
--num_train_epochs 3 \
--max_steps 8144 \
--warmup_steps 814 \
--do_lower_case \
--learning_rate 3e-5 \
--max_seq_length 512 \
--doc_stride 128 \
--save_steps 2000 \
--per_gpu_train_batch_size 1 \
--gradient_accumulation_steps 24 \
--output_dir ${MODEL_PATH}

CUDA_VISIBLE_DEVICES=0 python ${RUN_SQUAD_DIR}/run_squad.py \
--model_type albert \
--model_name_or_path ${MODEL_PATH} \
--do_eval \
--train_file ${SQUAD_DIR}/train-v2.0.json \
--predict_file ${SQUAD_DIR}/dev-v2.0.json \
--version_2_with_negative \
--do_lower_case \
--max_seq_length 512 \
--per_gpu_eval_batch_size 48 \
--output_dir ${MODEL_PATH}
```

### using the following system & software:

```
OS/Platform: Linux-4.15.0-76-generic-x86_64-with-debian-buster-sid
GPU/CPU: 2 x NVIDIA 1080Ti / Intel i7-8700
Transformers: 2.3.0
PyTorch: 1.4.0
TensorFlow: 2.1.0
Python: 3.7.6
```

### Inferencing / prediction works with the current Transformers v2.4.1

### Access this albert_xxlargev1_sqd2_512 fine-tuned model with "tried & true" code:

```python
config_class, model_class, tokenizer_class = \
        AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer

model_name_or_path = "ahotrod/albert_xxlargev1_squad2_512"
config = config_class.from_pretrained(model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
model = model_class.from_pretrained(model_name_or_path, config=config)
```

### or the AutoModels (AutoConfig, AutoTokenizer & AutoModel) should also work, however I have yet to use them in my app & confirm:

```python
from transformers import AutoConfig, AutoTokenizer, AutoModel

model_name_or_path = "ahotrod/albert_xxlargev1_squad2_512"
config = AutoConfig.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
model = AutoModel.from_pretrained(model_name_or_path, config=config)
```