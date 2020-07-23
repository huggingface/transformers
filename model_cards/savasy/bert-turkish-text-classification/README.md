---
language: tr
---

# Turkish Text Classification

This model is a fine-tune model of https://github.com/stefan-it/turkish-bert by using text classification data where there are 7 categories as follows

```
code_to_label={
 'LABEL_0': 'dunya ',
 'LABEL_1': 'ekonomi ',
 'LABEL_2': 'kultur ',
 'LABEL_3': 'saglik ',
 'LABEL_4': 'siyaset ',
 'LABEL_5': 'spor ',
 'LABEL_6': 'teknoloji '}
 
 ```


## Data 
The following Turkish benchmark dataset is used for fine-tuning

https://www.kaggle.com/savasy/ttc4900

## Quick Start

Bewgin with installing transformers as follows
> pip install transformers

```
# Code:
# import libraries
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification
tokenizer= AutoTokenizer.from_pretrained("savasy/bert-turkish-text-classification")

# build and load model, it take time depending on your internet connection
model= AutoModelForSequenceClassification.from_pretrained("savasy/bert-turkish-text-classification")

# make pipeline
nlp=pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# apply model
nlp("bla bla")
# [{'label': 'LABEL_2', 'score': 0.4753005802631378}]

code_to_label={
 'LABEL_0': 'dunya ',
 'LABEL_1': 'ekonomi ',
 'LABEL_2': 'kultur ',
 'LABEL_3': 'saglik ',
 'LABEL_4': 'siyaset ',
 'LABEL_5': 'spor ',
 'LABEL_6': 'teknoloji '}
 
code_to_label[nlp("bla bla")[0]['label']]
# > 'kultur '
```

## How the model was trained

```

## loading data for Turkish text classification
import pandas as pd
# https://www.kaggle.com/savasy/ttc4900
df=pd.read_csv("7allV03.csv")
df.columns=["labels","text"]
df.labels=pd.Categorical(df.labels)

traind_df=...
eval_df=...

# model
from simpletransformers.classification import ClassificationModel
import torch,sklearn

model_args = {
    "use_early_stopping": True,
    "early_stopping_delta": 0.01,
    "early_stopping_metric": "mcc",
    "early_stopping_metric_minimize": False,
    "early_stopping_patience": 5,
    "evaluate_during_training_steps": 1000,
    "fp16": False,
    "num_train_epochs":3
}

model = ClassificationModel(
    "bert", 
    "dbmdz/bert-base-turkish-cased",
     use_cuda=cuda_available, 
     args=model_args, 
     num_labels=7
)
model.train_model(train_df, acc=sklearn.metrics.accuracy_score)
```
For other training models please check https://simpletransformers.ai/


For the detailed usage of Turkish Text Classification please check [python notebook](https://github.com/savasy/TurkishTextClassification/blob/master/Bert_base_Text_Classification_for_Turkish.ipynb)
