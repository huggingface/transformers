This model is [Distilroberta base](https://huggingface.co/distilroberta-base) trained on SQuAD v2 as:

```
export SQUAD_DIR=../../squad2
python3 run_squad.py 
    --model_type robberta 
    --model_name_or_path distilroberta-base 
    --do_train 
    --do_eval 
    --overwrite_cache 
    --do_lower_case 
    --version_2_with_negative 
    --save_steps 100000 
    --train_file $SQUAD_DIR/train-v2.0.json 
    --predict_file $SQUAD_DIR/dev-v2.0.json 
    --per_gpu_train_batch_size 8 
    --num_train_epochs 3 
    --learning_rate 3e-5 
    --max_seq_length 384 
    --doc_stride 128 
    --output_dir ./tmp/distilroberta_fine_tuned/
```

Performance on a dev subset is close to the original paper:

```
Results: 
{
    'exact': 70.9279368213228, 
    'f1': 74.60439802429168, 
    'total': 6078, 
    'HasAns_exact': 67.62886597938144, 
    'HasAns_f1': 75.30774267754136, 
    'HasAns_total': 2910, 
    'NoAns_exact': 73.95833333333333, 
    'NoAns_f1': 73.95833333333333, 'NoAns_total': 3168, 
    'best_exact': 70.94438960184272, 
    'best_exact_thresh': 0.0, 
    'best_f1': 74.62085080481161, 
    'best_f1_thresh': 0.0
}
```

We are hopeful this might save you time, energy, and compute. Cheers!