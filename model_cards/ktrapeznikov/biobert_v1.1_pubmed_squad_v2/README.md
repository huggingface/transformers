### Model
**[`monologg/biobert_v1.1_pubmed`](https://huggingface.co/monologg/biobert_v1.1_pubmed)** fine-tuned on **[`SQuAD V2`](https://rajpurkar.github.io/SQuAD-explorer/)** using **[`run_squad.py`](https://github.com/huggingface/transformers/blob/master/examples/run_squad.py)**

This model is cased.

### Training Parameters
Trained on 4 NVIDIA GeForce RTX 2080 Ti 11Gb
```bash
BASE_MODEL=monologg/biobert_v1.1_pubmed
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
| exact             | 75.97068980038743 |
| f1                | 79.37043950121722 |
| total             | 11873.0           |
| HasAns_exact      | 74.13967611336032 |
| HasAns_f1         | 80.94892513460755 |
| HasAns_total      | 5928.0            |
| NoAns_exact       | 77.79646761984861 |
| NoAns_f1          | 77.79646761984861 |
| NoAns_total       | 5945.0            |
| best_exact        | 75.97068980038743 |
| best_exact_thresh | 0.0               |
| best_f1           | 79.37043950121729 |
| best_f1_thresh    | 0.0               |


### Usage

See [huggingface documentation](https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering). Training on `SQuAD V2` allows the model to score if a paragraph contains an answer:
```python
start_scores, end_scores = model(input_ids) 
span_scores = start_scores.softmax(dim=1).log()[:,:,None] + end_scores.softmax(dim=1).log()[:,None,:]
ignore_score = span_scores[:,0,0] #no answer scores
    
```

