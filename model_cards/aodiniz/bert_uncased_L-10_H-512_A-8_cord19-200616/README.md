# BERT L-10 H-512 fine-tuned on MLM (CORD-19 2020/06/16)

BERT model with [10 Transformer layers and hidden embedding of size 512](https://huggingface.co/google/bert_uncased_L-10_H-512_A-8), referenced in [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962), fine-tuned for MLM on CORD-19 dataset (as released on 2020/06/16).

## Training the model

```bash
python run_language_modeling.py
    --model_type bert
    --model_name_or_path google/bert_uncased_L-10_H-512_A-8
    --do_train
    --train_data_file {cord19-200616-dataset}
    --mlm
    --mlm_probability 0.2
    --line_by_line
    --block_size 512
    --per_device_train_batch_size 10
    --learning_rate 3e-5
    --num_train_epochs 2
    --output_dir bert_uncased_L-10_H-512_A-8_cord19-200616
