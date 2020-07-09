This model is [ALBERT base v2](https://huggingface.co/albert-base-v2) trained on SQuAD v2 as:

```
export SQUAD_DIR=../../squad2
python3 run_squad.py 
    --model_type albert 
    --model_name_or_path albert-base-v2 
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
    --output_dir ./tmp/albert_fine/
```

Performance on a dev subset is close to the original paper:

```
Results: 
{
    'exact': 78.71010200723923, 
    'f1': 81.89228117126069, 
    'total': 6078, 
    'HasAns_exact': 75.39518900343643, 
    'HasAns_f1': 82.04167868004215, 
    'HasAns_total': 2910, 
    'NoAns_exact': 81.7550505050505, 
    'NoAns_f1': 81.7550505050505, 
    'NoAns_total': 3168, 
    'best_exact': 78.72655478775913, 
    'best_exact_thresh': 0.0, 
    'best_f1': 81.90873395178066, 
    'best_f1_thresh': 0.0
}
```

We are hopeful this might save you time, energy, and compute. Cheers!