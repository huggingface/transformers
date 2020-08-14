### Model
**[`albert-xlarge-v2`](https://huggingface.co/albert-xlarge-v2)** fine-tuned on **[`SQuAD V2`](https://rajpurkar.github.io/SQuAD-explorer/)** using **[`run_squad.py`](https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_squad.py)**

### Training Parameters
Trained on 4 NVIDIA GeForce RTX 2080 Ti 11Gb
```bash
BASE_MODEL=albert-xlarge-v2
python run_squad.py \
  --version_2_with_negative \
  --model_type albert \
  --model_name_or_path $BASE_MODEL \
  --output_dir $OUTPUT_MODEL \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --per_gpu_train_batch_size 3 \
  --per_gpu_eval_batch_size 64 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps 2000 \
  --threads 24 \
  --warmup_steps 814 \
  --gradient_accumulation_steps 4 \
  --fp16 \
  --do_train
```
  
### Evaluation

Evaluation on the dev set. I did not sweep for best threshold.

|                   | val               |
|-------------------|-------------------|
| exact             | 84.41842836688285 |
| f1                | 87.4628460501696  |
| total             | 11873.0           |
| HasAns_exact      | 80.68488529014844 |
| HasAns_f1         | 86.78245127423482 |
| HasAns_total      | 5928.0            |
| NoAns_exact       | 88.1412952060555  |
| NoAns_f1          | 88.1412952060555  |
| NoAns_total       | 5945.0            |
| best_exact        | 84.41842836688285 |
| best_exact_thresh | 0.0               |
| best_f1           | 87.46284605016956 |
| best_f1_thresh    | 0.0               |


### Usage

See [huggingface documentation](https://huggingface.co/transformers/model_doc/albert.html#albertforquestionanswering). Training on `SQuAD V2` allows the model to score if a paragraph contains an answer:
```python
start_scores, end_scores = model(input_ids) 
span_scores = start_scores.softmax(dim=1).log()[:,:,None] + end_scores.softmax(dim=1).log()[:,None,:]
ignore_score = span_scores[:,0,0] #no answer scores
    
```

