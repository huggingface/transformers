# BERT-Small fine-tuned on CORD-19 dataset

[BERT L6_H-512_A-8 model](https://huggingface.co/google/bert_uncased_L-6_H-512_A-8) fine-tuned on the [CORD-19 dataset](https://www.semanticscholar.org/cord19).

## CORD-19 data subset
The training data for this dataset is stored as a [Kaggle dataset](https://www.kaggle.com/davidmezzetti/cord19-qa?select=cord19.txt). The training
data is a subset of the full corpus, focusing on high-quality, study-design detected articles.

## Building the model

```bash
python run_language_modeling.py
    --model_type bert
    --model_name_or_path google/bert_uncased_L-6_H-512_A-8
    --do_train
    --mlm
    --line_by_line
    --block_size 512
    --train_data_file cord19.txt
    --per_gpu_train_batch_size 4
    --learning_rate 3e-5
    --num_train_epochs 3.0
    --output_dir bert-small-cord19
    --save_steps 0
    --overwrite_output_dir
