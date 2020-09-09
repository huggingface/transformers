---
language:
- es
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

# BERT codemixed base model for spanglish (cased)

This model was built using [lingualytics](https://github.com/lingualytics/py-lingualytics), an open-source library that supports code-mixed analytics.

## Model description

Input for the model: Any codemixed spanglish text
Output for the model: Sentiment. (0 - Negative, 1 - Neutral, 2 - Positive)

I took a bert-base-multilingual-cased model from Huggingface and finetuned it on [CS-EN-ES-CORPUS](http://www.grupolys.org/software/CS-CORPORA/cs-en-es-corpus-wassa2015.txt) dataset.  

Performance of this model on the dataset

| metric     |    score |
|------------|----------|
| acc        | 0.718615 |
| f1         | 0.71759 |
| acc_and_f1 | 0.718103 |
| precision  | 0.719302 |
| recall     | 0.718615 |

## Intended uses & limitations

Make sure to preprocess your data using [these methods](https://github.com/microsoft/GLUECoS/blob/master/Data/Preprocess_Scripts/preprocess_sent_en_es.py) before using this model.

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

#### Limitations and bias

Since I dont know spanish, I cant verify the quality of annotations or the dataset itself. This is a very simple transfer learning approach and I'm open to discussions to improve upon this.

## Training data

I trained on the dataset on the [bert-base-multilingual-cased model](https://huggingface.co/bert-base-multilingual-cased).

## Training procedure

Followed the preprocessing techniques followed [here](https://github.com/microsoft/GLUECoS/blob/master/Data/Preprocess_Scripts/preprocess_sent_en_es.py)

## Eval results

### BibTeX entry and citation info

```bibtex
@inproceedings{khanuja-etal-2020-gluecos,
    title = "{GLUEC}o{S}: An Evaluation Benchmark for Code-Switched {NLP}",
    author = "Khanuja, Simran  and
      Dandapat, Sandipan  and
      Srinivasan, Anirudh  and
      Sitaram, Sunayana  and
      Choudhury, Monojit",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.329",
    pages = "3575--3585"
}
```
