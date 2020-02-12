---
language: spanish
thumbnail: https://i.imgur.com/jgBdimh.png
---

# BETO (Spanish BERT) + Spanish SQuAD2.0

This model is provided by [BETO team](https://github.com/dccuchile/beto) and fine-tuned on [SQuAD-es-v2.0](https://github.com/ccasimiro88/TranslateAlignRetrieve) for **Q&A** downstream task.

## Details of the language model('dccuchile/bert-base-spanish-wwm-cased')

Language model ([**'dccuchile/bert-base-spanish-wwm-cased'**](https://github.com/dccuchile/beto/blob/master/README.md)):

BETO is a [BERT model](https://github.com/google-research/bert) trained on a [big Spanish corpus](https://github.com/josecannete/spanish-corpora). BETO is of size similar to a BERT-Base and was trained with the Whole Word Masking technique. Below you find Tensorflow and Pytorch checkpoints for the uncased and cased versions, as well as some results for Spanish benchmarks comparing BETO with [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md) as well as other (not BERT-based) models.

## Details of the downstream task (Q&A) - Dataset
[SQuAD-es-v2.0](https://github.com/ccasimiro88/TranslateAlignRetrieve)

| Dataset                | # Q&A |
| ---------------------- | ----- |
| SQuAD2.0 Train         | 130 K |
| SQuAD2.0-es-v2.0       | 111 K |
| SQuAD2.0 Dev           | 12  K |
| SQuAD-es-v2.0-small Dev| 69  K |

## Model training

The model was trained on a Tesla P100 GPU and 25GB of RAM with the following command:

```bash
export SQUAD_DIR=path/to/nl_squad
python transformers/examples/run_squad.py \
  --model_type bert \
  --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train_nl-v2.0.json \
  --predict_file $SQUAD_DIR/dev_nl-v2.0.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /content/model_output \
  --save_steps 5000 \
  --threads 4 \
  --version_2_with_negative 
```

## Results:


  | Metric               | # Value |
| ---------------------- | ----- |
| **Exact**              | **76.50**50 |
| **F1**                 | **86.07**81 |

```json
{
  "exact": 76.50501430594491,
  "f1": 86.07818773108252,
  "total": 69202,
  "HasAns_exact": 67.93020719738277,
  "HasAns_f1": 82.37912207996466,
  "HasAns_total": 45850,
  "NoAns_exact": 93.34104145255225,
  "NoAns_f1": 93.34104145255225,
  "NoAns_total": 23352,
  "best_exact": 76.51223953064941,
  "best_exact_thresh": 0.0,
  "best_f1": 86.08541295578848,
  "best_f1_thresh": 0.0
}
```

### Model in action (in a Colab Notebook)
<details>

1.  Set the context and ask some questions:

![Set context and questions](https://media.giphy.com/media/mCIaBpfN0LQcuzkA2F/giphy.gif)

2.  Run predictions:

![Run the model](https://media.giphy.com/media/WT453aptcbCP7hxWTZ/giphy.gif)
</details>

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
