---
language: spanish
thumbnail: https://i.imgur.com/jgBdimh.png
---

# Spanish BERT (BETO) + POS

This model is a fine-tuned on [NER-C](https://www.kaggle.com/nltkdata/conll-corpora) Of the Spanish BERT cased [(BETO)](https://github.com/dccuchile/beto) for **POS** (Part of Speech tagging) downstream task.

## Details of the downstream task (POS) - Dataset

- [Dataset:  CONLL Corpora ES](https://www.kaggle.com/nltkdata/conll-corpora) with data augmentation techniques

I preprocessed the dataset and splitted it as train / dev (80/20)

| Dataset                | # Examples |
| ---------------------- | ----- |
| Train                  | 340 K |
| Dev                    | 50 K |


- [Fine-tune on NER script provided by Huggingface](https://github.com/huggingface/transformers/blob/master/examples/run_ner.py)

- Labels covered:

```
AO, AQ, CC, CS, DA, DD, DE, DI, DN, DP, DT, Faa, Fat, Fc, Fd, Fe, Fg, Fh, Fia, Fit, Fp, Fpa, Fpt, Fs, Ft, Fx, Fz, I, NC, NP, P0, PD, PI, PN, PP, PR, PT, PX, RG, RN, SP, VAI, VAM, VAN, VAP, VAS, VMG, VMI, VMM, VMN, VMP, VMS, VSG, VSI, VSM, VSN, VSP, VSS, Y and Z
```


## Metrics on evaluation set:

|                                                      Metric                                                       |  # score  |
| :------------------------------------------------------------------------------------: | :-------: |
| F1                                       | **90.06**  
| Precision                                | **89.46** | 
| Recall                                   | **90.67** |                                    

## Model in action

Fast usage with **pipelines**:

```python
from transformers import pipeline

nlp_pos = pipeline(
    "ner",
    model="mrm8488/bert-spanish-cased-finetuned-pos",
    tokenizer=(
        'mrm8488/bert-spanish-cased-finetuned-pos',  
        {"use_fast": False}
))


text = 'Mis amigos están pensando en viajar a Londres este verano'

nlp_pos(text)

#Output:
'''
[{'entity': 'NC', 'score': 0.7792173624038696, 'word': '[CLS]'},
 {'entity': 'DP', 'score': 0.9996283650398254, 'word': 'Mis'},
 {'entity': 'NC', 'score': 0.9999253749847412, 'word': 'amigos'},
 {'entity': 'VMI', 'score': 0.9998560547828674, 'word': 'están'},
 {'entity': 'VMG', 'score': 0.9992249011993408, 'word': 'pensando'},
 {'entity': 'SP', 'score': 0.9999602437019348, 'word': 'en'},
 {'entity': 'VMN', 'score': 0.9998666048049927, 'word': 'viajar'},
 {'entity': 'SP', 'score': 0.9999545216560364, 'word': 'a'},
 {'entity': 'VMN', 'score': 0.8722310662269592, 'word': 'Londres'},
 {'entity': 'DD', 'score': 0.9995203614234924, 'word': 'este'},
 {'entity': 'NC', 'score': 0.9999248385429382, 'word': 'verano'},
 {'entity': 'NC', 'score': 0.8802427649497986, 'word': '[SEP]'}]
 '''
```
![model in action](https://media.giphy.com/media/jVC9m1cNrdIWuAAtjy/giphy.gif)


> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
