# BERT-Small CORD-19 fine-tuned on SQuAD 2.0

[bert-small-cord19 model](https://huggingface.co/NeuML/bert-small-cord19) fine-tuned on SQuAD 2.0

## Building the model

```bash
python run_squad.py
    --model_type bert
    --model_name_or_path bert-small-cord19
    --do_train
    --do_eval
    --do_lower_case
    --version_2_with_negative
    --train_file train-v2.0.json
    --predict_file dev-v2.0.json
    --per_gpu_train_batch_size 8
    --learning_rate 3e-5
    --num_train_epochs 3.0
    --max_seq_length 384
    --doc_stride 128
    --output_dir bert-small-cord19-squad2
    --save_steps 0
    --threads 8
    --overwrite_cache
    --overwrite_output_dir
