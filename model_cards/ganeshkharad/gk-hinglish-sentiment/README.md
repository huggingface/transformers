---
language:
- hi-en

tags:
- sentiment
- multilingual
- hindi codemix
- hinglish 
license: apache-2.0
datasets:
- sail
---

# Sentiment Classification for hinglish text: `gk-hinglish-sentiment`

## Model description

Trained small amount of reviews dataset

## Intended uses & limitations

I wanted something to work well with hinglish data as it is being used in India mostly.
The training data was not much as expected

#### How to use

```python
#sample code 
from transformers import BertTokenizer, BertForSequenceClassification
tokenizerg = BertTokenizer.from_pretrained("/content/model")
modelg = BertForSequenceClassification.from_pretrained("/content/model")

text = "kuch bhi type karo hinglish mai"
encoded_input = tokenizerg(text, return_tensors='pt')
output = modelg(**encoded_input)
print(output)
#output contains 3 lables LABEL_0 = Negative ,LABEL_1 = Nuetral ,LABEL_2 = Positive
```

#### Limitations and bias

The data contains only hinglish codemixed text it and was very much limited may be I will Update this model if I can get good amount of data

## Training data

Training data contains labeled data for 3 labels

link to the pre-trained model card with description of the pre-training data.
I have Tuned below model

https://huggingface.co/rohanrajpal/bert-base-multilingual-codemixed-cased-sentiment


### BibTeX entry and citation info

```@inproceedings{khanuja-etal-2020-gluecos,
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
