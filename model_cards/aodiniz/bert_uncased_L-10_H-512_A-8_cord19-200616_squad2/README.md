---
datasets:
- squad_v2
---

# BERT L-10 H-512 CORD-19 (2020/06/16) fine-tuned on SQuAD v2.0

BERT model with [10 Transformer layers and hidden embedding of size 512](https://huggingface.co/google/bert_uncased_L-10_H-512_A-8), referenced in [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962), [fine-tuned for MLM](https://huggingface.co/aodiniz/bert_uncased_L-10_H-512_A-8_cord19-200616) on CORD-19 dataset (as released on 2020/06/16) and fine-tuned for QA on SQuAD v2.0.

## Training the model

```bash
python run_squad.py
    --model_type bert
    --model_name_or_path aodiniz/bert_uncased_L-10_H-512_A-8_cord19-200616
    --train_file 'train-v2.0.json'
    --predict_file 'dev-v2.0.json'
    --do_train
    --do_eval
    --do_lower_case
    --version_2_with_negative
    --max_seq_length 384
    --per_gpu_train_batch_size 10
    --learning_rate 3e-5
    --num_train_epochs 2
    --output_dir bert_uncased_L-10_H-512_A-8_cord19-200616_squad2
