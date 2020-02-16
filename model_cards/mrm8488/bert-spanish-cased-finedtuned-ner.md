---
language: spanish
thumbnail: https://i.imgur.com/jgBdimh.png
---

# Spanish BERT (BETO) + NER

This model is a fine-tuned on [NER-C](https://www.kaggle.com/nltkdata/conll-corpora) of the Spanish BERT cased [(BETO)](https://github.com/dccuchile/beto) for **NER** downstream task.

## Details of the downstream task (NER) - Dataset

- [Dataset](https://www.kaggle.com/nltkdata/conll-corpora)
- [Fine-tune on NER script](https://github.com/huggingface/transformers/blob/master/examples/run_ner.py)

## Comparison:

|                                                      Model                                                       |  # score  |
| :--------------------------------------------------------------------------------------------------------------: | :-------: |
|                                        bert-base-spanish-wwm-cased (BETO)                                        |   86.07   |
| [bert-spanish-cased-finedtuned-ner (this one)](https://huggingface.co/mrm8488/bert-spanish-cased-finedtuned-ner) | **91.66** |
|                                              Best Multilingual BERT                                              |   87.38   |

```
 ***** All metrics on Eval results  *****

f1 = 0.9166444740346205
loss = 0.09283923702787433
precision = 0.9159127195316658
recall = 0.9173773987206824
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
