---
language: ar
tags:
- qarib

license: apache-2.0
datasets:
- Arabic GigaWord
- Abulkhair Arabic Corpus
- opus
- Twitter data
---

# QARiB: QCRI Arabic and Dialectal BERT

## About QARiB
QCRI Arabic and Dialectal BERT  (QARiB) model, was trained on a collection of ~ 420 Million tweets and ~ 180 Million sentences of text.
For Tweets, the data was collected using twitter API and using language filter. `lang:ar`. For Text data, it was a combination from 
[Arabic GigaWord](url), [Abulkhair Arabic Corpus]() and [OPUS](http://opus.nlpl.eu/).

### bert-base-qarib60_860k
- Data size: 60Gb
- Number of Iterations: 860k
- Loss: 2.2454472

## Training QARiB
The training of the model has been performed using Google’s original Tensorflow code on Google Cloud TPU v2.
We used a Google Cloud Storage bucket, for persistent storage of training data and models.
See more details in [Training QARiB](../Training_QARiB.md)

## Using QARiB

You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to be fine-tuned on a downstream task. See the model hub to look for fine-tuned versions on a task that interests you. For more details, see [Using QARiB](../Using_QARiB.md)

### How to use
You can use this model directly with a pipeline for masked language modeling:

```python
>>>from transformers import pipeline
>>>fill_mask = pipeline("fill-mask", model="./models/data60gb_86k")

>>> fill_mask("شو عندكم يا [MASK]")
[{'sequence': '[CLS] شو عندكم يا عرب [SEP]', 'score': 0.0990147516131401, 'token': 2355, 'token_str': 'عرب'}, 
{'sequence': '[CLS] شو عندكم يا جماعة [SEP]', 'score': 0.051633741706609726, 'token': 2308, 'token_str': 'جماعة'}, 
{'sequence': '[CLS] شو عندكم يا شباب [SEP]', 'score': 0.046871256083250046, 'token': 939, 'token_str': 'شباب'}, 
{'sequence': '[CLS] شو عندكم يا رفاق [SEP]', 'score': 0.03598872944712639, 'token': 7664, 'token_str': 'رفاق'}, 
{'sequence': '[CLS] شو عندكم يا ناس [SEP]', 'score': 0.031996358186006546, 'token': 271, 'token_str': 'ناس'}]

>>> fill_mask("قللي وشفيييك يرحم [MASK]")
[{'sequence': '[CLS] قللي وشفيييك يرحم والديك [SEP]', 'score': 0.4152909517288208, 'token': 9650, 'token_str': 'والديك'}, 
{'sequence': '[CLS] قللي وشفيييك يرحملي [SEP]', 'score': 0.07663793861865997, 'token': 294, 'token_str': '##لي'}, 
{'sequence': '[CLS] قللي وشفيييك يرحم حالك [SEP]', 'score': 0.0453166700899601, 'token': 2663, 'token_str': 'حالك'}, 
{'sequence': '[CLS] قللي وشفيييك يرحم امك [SEP]', 'score': 0.04390475153923035, 'token': 1942, 'token_str': 'امك'}, 
{'sequence': '[CLS] قللي وشفيييك يرحمونك [SEP]', 'score': 0.027349254116415977, 'token': 3283, 'token_str': '##ونك'}]

>>> fill_mask("وقام المدير [MASK]")
[
{'sequence': '[CLS] وقام المدير بالعمل [SEP]', 'score': 0.0678194984793663, 'token': 4230, 'token_str': 'بالعمل'}, 
{'sequence': '[CLS] وقام المدير بذلك [SEP]', 'score': 0.05191086605191231, 'token': 984, 'token_str': 'بذلك'}, 
{'sequence': '[CLS] وقام المدير بالاتصال [SEP]', 'score': 0.045264165848493576, 'token': 26096, 'token_str': 'بالاتصال'}, 
{'sequence': '[CLS] وقام المدير بعمله [SEP]', 'score': 0.03732728958129883, 'token': 40486, 'token_str': 'بعمله'}, 
{'sequence': '[CLS] وقام المدير بالامر [SEP]', 'score': 0.0246378555893898, 'token': 29124, 'token_str': 'بالامر'}
]
>>> fill_mask("وقامت المديرة [MASK]")

[{'sequence': '[CLS] وقامت المديرة بذلك [SEP]', 'score': 0.23992691934108734, 'token': 984, 'token_str': 'بذلك'}, 
{'sequence': '[CLS] وقامت المديرة بالامر [SEP]', 'score': 0.108805812895298, 'token': 29124, 'token_str': 'بالامر'}, 
{'sequence': '[CLS] وقامت المديرة بالعمل [SEP]', 'score': 0.06639821827411652, 'token': 4230, 'token_str': 'بالعمل'}, 
{'sequence': '[CLS] وقامت المديرة بالاتصال [SEP]', 'score': 0.05613093823194504, 'token': 26096, 'token_str': 'بالاتصال'}, 
{'sequence': '[CLS] وقامت المديرة المديرة [SEP]', 'score': 0.021778125315904617, 'token': 41635, 'token_str': 'المديرة'}]
```
## Training procedure

The training of the model has been performed using Google’s original Tensorflow code on eight core Google Cloud TPU v2.
We used a Google Cloud Storage bucket, for persistent storage of training data and models.

## Eval results

We evaluated QARiB models on five NLP downstream task:
- Sentiment Analysis
- Emotion Detection
- Named-Entity Recognition (NER)
- Offensive Language Detection
- Dialect Identification

The results obtained from QARiB models outperforms multilingual BERT/AraBERT/ArabicBERT.


## Model Weights and Vocab Download
TBD

## Contacts

Ahmed Abdelali, Sabit Hassan, Hamdy Mubarak, Kareem Darwish and Younes Samih


