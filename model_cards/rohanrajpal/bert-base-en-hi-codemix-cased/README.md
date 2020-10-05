---
language:
- hi
- en
tags:
- es
- en
- codemix
license: "apache-2.0"
datasets:
- SAIL 2017
metrics:
- fscore
- accuracy
- precision
- recall
---

# BERT codemixed base model for Hinglish (cased)

This model was built using [lingualytics](https://github.com/lingualytics/py-lingualytics), an open-source library that supports code-mixed analytics.

## Model description

Input for the model: Any codemixed Hinglish text
Output for the model: Sentiment. (0 - Negative, 1 - Neutral, 2 - Positive)

I took a bert-base-multilingual-cased model from Huggingface and finetuned it on [SAIL 2017](http://www.dasdipankar.com/SAILCodeMixed.html) dataset.  

## Eval results

Performance of this model on the dataset

| metric     |    score |
|------------|----------|
| acc        | 0.55873 |
| f1         | 0.558369 |
| acc_and_f1 | 0.558549 |
| precision  | 0.558075 |
| recall     | 0.55873 |

#### How to use

Here is how to use this model to get the features of a given text in *PyTorch*:

```python
# You can include sample code which will be formatted
from transformers import BertTokenizer, BertModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('rohanrajpal/bert-base-en-es-codemix-cased')
model = AutoModelForSequenceClassification.from_pretrained('rohanrajpal/bert-base-en-es-codemix-cased')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in *TensorFlow*:

```python
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('rohanrajpal/bert-base-en-es-codemix-cased')
model = TFBertModel.from_pretrained('rohanrajpal/bert-base-en-es-codemix-cased')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

#### Preprocessing

Followed standard preprocessing techniques:
- removed digits
- removed punctuation
- removed stopwords
- removed excess whitespace
Here's the snippet

```python
from pathlib import Path
import pandas as pd
from lingualytics.preprocessing import remove_lessthan, remove_punctuation, remove_stopwords
from lingualytics.stopwords import hi_stopwords,en_stopwords
from texthero.preprocessing import remove_digits, remove_whitespace

root = Path('<path-to-data>')

for file in 'test','train','validation':
  tochange = root / f'{file}.txt'
  df = pd.read_csv(tochange,header=None,sep='\t',names=['text','label'])
  df['text'] = df['text'].pipe(remove_digits) \
                                    .pipe(remove_punctuation) \
                                    .pipe(remove_stopwords,stopwords=en_stopwords.union(hi_stopwords)) \
                                    .pipe(remove_whitespace)
  df.to_csv(tochange,index=None,header=None,sep='\t')
```

## Training data

The dataset and annotations are not good, but this is the best dataset I could find. I am working on procuring my own dataset and will try to come up with a better model!

## Training procedure

I trained on the dataset on the [bert-base-multilingual-cased model](https://huggingface.co/bert-base-multilingual-cased).
