This model is [BERT base uncased](https://huggingface.co/bert-base-uncased) trained on SQuAD v2 as:

```
export SQUAD_DIR=../../squad2
python3 run_squad.py 
    --model_type bert 
    --model_name_or_path bert-base-uncased 
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
    --output_dir ./tmp/bert_fine_tuned/
```

Performance on a dev subset is close to the original paper:

```
Results: 
{
    'exact': 72.35932872655479, 
    'f1': 75.75355132564763, 
    'total': 6078, 
    'HasAns_exact': 74.29553264604812, 
    'HasAns_f1': 81.38490892002987, 
    'HasAns_total': 2910, 
    'NoAns_exact': 70.58080808080808, 
    'NoAns_f1': 70.58080808080808, 
    'NoAns_total': 3168, 
    'best_exact': 72.35932872655479, 
    'best_exact_thresh': 0.0, 
    'best_f1': 75.75355132564766, 
    'best_f1_thresh': 0.0
}
```

We are hopeful this might save you time, energy, and compute. Cheers!