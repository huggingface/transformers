---
language: spanish
thumbnail: https://i.imgur.com/jgBdimh.png
---

# Spanish BERT (BETO) + NER

This model is a fine-tuned on [NER-C](https://www.kaggle.com/nltkdata/conll-corpora) of the Spanish BERT cased [(BETO)](https://github.com/dccuchile/beto) for **NER** downstream task.

## Details of the downstream task (NER) - Dataset

- [Dataset:  CONLL Corpora ES](https://www.kaggle.com/nltkdata/conll-corpora) 

I preprocessed the dataset and splitted it as train / dev (80/20)

| Dataset                | # Examples |
| ---------------------- | ----- |
| Train                  | 8.7 K |
| Dev                    | 2.2 K |


- [Fine-tune on NER script](https://github.com/huggingface/transformers/blob/master/examples/run_ner.py)

```bash
!export NER_DIR='/content/ner_dataset'
!python /content/transformers/examples/run_ner.py \
  --model_type bert \
  --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
  --do_train \
  --do_eval \
  --data_dir '/content/ner_dataset' \
  --num_train_epochs 15.0 \
  --max_seq_length 384 \
  --output_dir /content/model_output \
  --save_steps 5000 \

```

## Comparison:

|                                                      Model                                                       |  # score  |
| :--------------------------------------------------------------------------------------------------------------: | :-------: |
|                                        bert-base-spanish-wwm-cased (BETO)                                        |   88.43   |
| [bert-spanish-cased-finetuned-ner (this one)](https://huggingface.co/mrm8488/bert-spanish-cased-finetuned-ner) | **89.65** |
|                                              Best Multilingual BERT                                              |   87.38   |

```
 ***** All metrics on Eval results  *****

f1 = 0.8965040489828165
loss = 0.11504213575173258
precision = 0.893679858239811
recall = 0.8993461462254805
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
