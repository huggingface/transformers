This model is [Distilbert base uncased](https://huggingface.co/distilbert-base-uncased) trained on SQuAD v2 as:

```
export SQUAD_DIR=../../squad2
python3 run_squad.py 
    --model_type distilbert 
    --model_name_or_path distilbert-base-uncased
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
    --output_dir ./tmp/distilbert_fine_tuned/
```

Performance on a dev subset is close to the original paper:

```
Results: 
{
    'exact': 64.88976637051661, 
    'f1': 68.1776176526635, 
    'total': 6078, 
    'HasAns_exact': 69.7594501718213, 
    'HasAns_f1': 76.62665295288285, 
    'HasAns_total': 2910, 
    'NoAns_exact': 60.416666666666664, 
    'NoAns_f1': 60.416666666666664, 
    'NoAns_total': 3168, 
    'best_exact': 64.88976637051661, 
    'best_exact_thresh': 0.0, 
    'best_f1': 68.17761765266337, 
    'best_f1_thresh': 0.0
}
```

We are hopeful this might save you time, energy, and compute. Cheers!