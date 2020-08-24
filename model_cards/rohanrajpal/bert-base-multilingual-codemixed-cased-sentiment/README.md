---
language:
- hi
- en
tags:
- hi
- en
- codemix
license: "apache-2.0"
datasets:
- SAIL 2017
metrics:
- fscore
- accuracy
---

# BERT codemixed base model for hinglish (cased)

## Model description

Input for the model: Any codemixed hinglish text
Output for the model: Sentiment. (0 - Negative, 1 - Neutral, 2 - Positive)

I took a bert-base-multilingual-cased model from Huggingface and finetuned it on [SAIL 2017](http://www.dasdipankar.com/SAILCodeMixed.html) dataset.  

Performance of this model on the SAIL 2017 dataset

| metric     |    score |
|------------|----------|
| acc        | 0.588889 |
| f1         | 0.582678 |
| acc_and_f1 | 0.585783 |
| precision  | 0.586516 |
| recall     | 0.588889 |

## Intended uses & limitations

#### How to use

Here is how to use this model to get the features of a given text in *PyTorch*:

```python
# You can include sample code which will be formatted
from transformers import BertTokenizer, BertModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("rohanrajpal/bert-base-codemixed-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("rohanrajpal/bert-base-codemixed-uncased-sentiment")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in *TensorFlow*:

```python
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('rohanrajpal/bert-base-codemixed-uncased-sentiment')
model = TFBertModel.from_pretrained("rohanrajpal/bert-base-codemixed-uncased-sentiment")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

#### Limitations and bias

Coming soon!

## Training data

I trained on the SAIL 2017 dataset [link](http://amitavadas.com/SAIL/Data/SAIL_2017.zip) on this [pretrained model](https://huggingface.co/bert-base-multilingual-cased).

## Training procedure

No preprocessing.

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
