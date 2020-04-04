### Model
**[`allenai/scibert_scivocab_uncased`](https://huggingface.co/allenai/scibert_scivocab_uncased)** fine-tuned on **[`SQuAD V2`](https://rajpurkar.github.io/SQuAD-explorer/)** using **[`run_squad.py`](https://github.com/huggingface/transformers/blob/master/examples/run_squad.py)**

### Training Parameters
Trained on 4 NVIDIA GeForce RTX 2080 Ti 11Gb
```bash
BASE_MODEL=allenai/scibert_scivocab_uncased
python run_squad.py \
  --version_2_with_negative \
  --model_type albert \
  --model_name_or_path $BASE_MODEL \
  --output_dir $OUTPUT_MODEL \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --per_gpu_train_batch_size 18 \
  --per_gpu_eval_batch_size 64 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps 2000 \
  --threads 24 \
  --warmup_steps 550 \
  --gradient_accumulation_steps 1 \
  --fp16 \
  --logging_steps 50 \
  --do_train
```
  
### Evaluation

Evaluation on the dev set. I did not sweep for best threshold.

|                   | val               |
|-------------------|-------------------|
| exact             | 75.07790785816559 |
| f1                | 78.47735207283013 |
| total             | 11873.0           |
| HasAns_exact      | 70.76585695006747 |
| HasAns_f1         | 77.57449412292718 |
| HasAns_total      | 5928.0            |
| NoAns_exact       | 79.37762825904122 |
| NoAns_f1          | 79.37762825904122 |
| NoAns_total       | 5945.0            |
| best_exact        | 75.08633032931863 |
| best_exact_thresh | 0.0               |
| best_f1           | 78.48577454398324 |
| best_f1_thresh    | 0.0               |

### Usage

See [huggingface documentation](https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering). Training on `SQuAD V2` allows the model to score if a paragraph contains an answer:
```python
start_scores, end_scores = model(input_ids) 
span_scores = start_scores.softmax(dim=1).log()[:,:,None] + end_scores.softmax(dim=1).log()[:,None,:]
ignore_score = span_scores[:,0,0] #no answer scores
    
```

