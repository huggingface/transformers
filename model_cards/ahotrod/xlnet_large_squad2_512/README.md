## XLNet large language model fine-tuned on SQuAD2.0

### with the following results:

```
  "exact": 82.07698138633876,
  "f1": 85.898874470488,
  "total": 11873,
  "HasAns_exact": 79.60526315789474,
  "HasAns_f1": 87.26000954590184,
  "HasAns_total": 5928,
  "NoAns_exact": 84.54163162321278,
  "NoAns_f1": 84.54163162321278,
  "NoAns_total": 5945,
  "best_exact": 83.22243746315169,
  "best_exact_thresh": -11.112004280090332,
  "best_f1": 86.88541353813282,
  "best_f1_thresh": -11.112004280090332
```
### from script:
```
python -m torch.distributed.launch --nproc_per_node=2 ${RUN_SQUAD_DIR}/run_squad.py \
  --model_type xlnet \
  --model_name_or_path xlnet-large-cased \
  --do_train \
  --train_file ${SQUAD_DIR}/train-v2.0.json \
  --predict_file ${SQUAD_DIR}/dev-v2.0.json \
  --version_2_with_negative \
  --num_train_epochs 3 \
  --learning_rate 3e-5 \
  --adam_epsilon 1e-6 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --save_steps 2000 \
  --per_gpu_train_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --output_dir ${MODEL_PATH}

CUDA_VISIBLE_DEVICES=0 python ${RUN_SQUAD_DIR}/run_squad_II.py \
  --model_type xlnet \
  --model_name_or_path ${MODEL_PATH} \
  --do_eval \
  --train_file ${SQUAD_DIR}/train-v2.0.json \
  --predict_file ${SQUAD_DIR}/dev-v2.0.json \
  --version_2_with_negative \
  --max_seq_length 512 \
  --per_gpu_eval_batch_size 48 \
  --output_dir ${MODEL_PATH}
```
### using the following system & software:
```
OS/Platform: Linux-4.15.0-76-generic-x86_64-with-debian-buster-sid
GPU/CPU: 2 x NVIDIA 1080Ti / Intel i7-8700
Transformers: 2.1.1
PyTorch: 1.4.0
TensorFlow: 2.1.0
Python: 3.7.6
```
### Utilize this xlnet_large_squad2_512 fine-tuned model with:
```python
tokenizer = AutoTokenizer.from_pretrained("ahotrod/xlnet_large_squad2_512")
model = AutoModelForQuestionAnswering.from_pretrained("ahotrod/xlnet_large_squad2_512")
```
