# BERT L-4 H-256 fine-tuned on MLM (CORD-19 2020/06/16)

BERT model with [4 Transformer layers and hidden embedding of size 256](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4), referenced in [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962), fine-tuned for MLM on CORD-19 dataset (as released on 2020/06/16).

## Training the model

```bash
python run_language_modeling.py
    --model_type bert
    --model_name_or_path google/bert_uncased_L-4_H-256_A-4
    --do_train
    --train_data_file {cord19-200616-dataset}
    --mlm
    --mlm_probability 0.2
    --line_by_line
    --block_size 256
    --per_device_train_batch_size 20
    --learning_rate 3e-5
    --num_train_epochs 2
    --output_dir bert_uncased_L-4_H-256_A-4_cord19-200616
