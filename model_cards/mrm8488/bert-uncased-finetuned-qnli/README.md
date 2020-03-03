---
language: english
thumbnail:
---

# [BERT](https://huggingface.co/deepset/bert-base-cased-squad2) fine tuned on [QNLI](https://github.com/rhythmcao/QNLI)+ compression ([BERT-of-Theseus](https://github.com/JetRunner/BERT-of-Theseus))

I used a [Bert model fine tuned on **SQUAD v2**](https://huggingface.co/deepset/bert-base-cased-squad2) and then I fine tuned it on **QNLI** using **compression** (with a constant replacing rate) as proposed in **BERT-of-Theseus**

## Details of the downstream task (QNLI):

### Getting the dataset
```bash
wget https://raw.githubusercontent.com/rhythmcao/QNLI/master/data/QNLI/train.tsv
wget https://raw.githubusercontent.com/rhythmcao/QNLI/master/data/QNLI/test.tsv
wget https://raw.githubusercontent.com/rhythmcao/QNLI/master/data/QNLI/dev.tsv

mkdir QNLI_dataset
mv *.tsv QNLI_dataset
```

### Model training

The model was trained on a Tesla P100 GPU and 25GB of RAM with the following command:

```bash
!python /content/BERT-of-Theseus/run_glue.py \
  --model_name_or_path deepset/bert-base-cased-squad2 \
  --task_name qnli \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /content/QNLI_dataset \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --save_steps 2000 \
  --num_train_epochs 50 \
  --output_dir /content/ouput_dir \
  --evaluate_during_training \
  --replacing_rate 0.7 \
  --steps_for_replacing 2500 
```

## Metrics:

| Model          | Accuracy |
|-----------------|------|
| BERT-base       | 91.2 |
| BERT-of-Theseus | 88.8 |
| [bert-uncased-finetuned-qnli](https://huggingface.co/mrm8488/bert-uncased-finetuned-qnli) | 87.2
| DistillBERT     | 85.3 |




> [See all my models](https://huggingface.co/models?search=mrm8488)

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
