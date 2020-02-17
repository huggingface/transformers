---
language: spanish
thumbnail: https://i.imgur.com/jgBdimh.png
---

# BETO (Spanish BERT) + Spanish SQuAD2.0 + distillation using 'bert-base-multilingual-cased' as teacher

This model is a fine-tuned on [SQuAD-es-v2.0](https://github.com/ccasimiro88/TranslateAlignRetrieve) and **distilled** version of [BETO](https://github.com/dccuchile/beto) for **Q&A**.

Distillation makes the model **smaller, faster, cheaper and lighter** than [bert-base-spanish-wwm-cased-finetuned-spa-squad2-es](https://github.com/huggingface/transformers/blob/master/model_cards/mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es/README.md)

This model was fine-tuned on the same dataset but using **distillation** during the process as mentioned above (and one more train epoch).

The **teacher model** for the distillation was `bert-base-multilingual-cased`. It is the same teacher used for `distilbert-base-multilingual-cased` AKA [**DistilmBERT**](https://github.com/huggingface/transformers/tree/master/examples/distillation) (on average is twice as fast as **mBERT-base**).

## Details of the downstream task (Q&A) - Dataset

<details>

[SQuAD-es-v2.0](https://github.com/ccasimiro88/TranslateAlignRetrieve)

| Dataset                 | # Q&A |
| ----------------------- | ----- |
| SQuAD2.0 Train          | 130 K |
| SQuAD2.0-es-v2.0        | 111 K |
| SQuAD2.0 Dev            | 12 K  |
| SQuAD-es-v2.0-small Dev | 69 K  |

</details>

## Model training

The model was trained on a Tesla P100 GPU and 25GB of RAM with the following command:

```bash
!export SQUAD_DIR=/path/to/squad-v2_spanish \
&& python transformers/examples/distillation/run_squad_w_distillation.py \
  --model_type bert \
  --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
  --teacher_type bert \
  --teacher_name_or_path bert-base-multilingual-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v2.json \
  --predict_file $SQUAD_DIR/dev-v2.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /content/model_output \
  --save_steps 5000 \
  --threads 4 \
  --version_2_with_negative
```

## Results:

| Metric    | # Value     |
| --------- | ----------- |
| **Exact** | **90.77**48 |
| **F1**    | **94.94**71 |

```json
{
  "exact": 90.77483309730933,
  "f1": 94.94714391266254,
  "total": 69202,
  "HasAns_exact": 86.60850599781898,
  "HasAns_f1": 92.90582885592328,
  "HasAns_total": 45850,
  "NoAns_exact": 98.95512161699212,
  "NoAns_f1": 98.95512161699212,
  "NoAns_total": 23352,
  "best_exact": 90.77483309730933,
  "best_exact_thresh": 0.0,
  "best_f1": 94.94714391266305,
  "best_f1_thresh": 0.0
}
```

## Comparison:

|                              Model                              | f1 score  |
| :-------------------------------------------------------------: | :-------: |
|       bert-base-spanish-wwm-cased-finetuned-spa-squad2-es       |   86.07   |
| **distill**-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es | **94.94** |

So, yes, this version is even more accurate.

### Model in action (in a Colab Notebook)

<details>

1.  Set the context and ask some questions:

![Set context and questions](https://media.giphy.com/media/mCIaBpfN0LQcuzkA2F/giphy.gif)

2.  Run predictions:

![Run the model](https://media.giphy.com/media/WT453aptcbCP7hxWTZ/giphy.gif)

3. Using **Pipelines**

![qa-pipelines](https://media.giphy.com/media/L4YiWhinC3P1GF31t0/giphy.gif)

</details>

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
