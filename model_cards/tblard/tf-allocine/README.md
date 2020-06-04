---
language: french
---

# tf-allociné

A french sentiment analysis model, based on [CamemBERT](https://camembert-model.fr/), and finetuned on a large-scale dataset scraped from [Allociné.fr](http://www.allocine.fr/) user reviews.

## Results

| Validation Accuracy | Validation F1-Score | Test Accuracy | Test F1-Score |
|--------------------:| -------------------:| -------------:|--------------:|
|               97.39 |               97.36 |         97.44 |         97.34 |

The dataset and the evaluation code are available on [this repo](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert).

## Usage

```python
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")
model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

print(nlp("Alad'2 est clairement le meilleur film de l'année 2018.")) # POSITIVE
print(nlp("Juste whoaaahouuu !")) # POSITIVE
print(nlp("NUL...A...CHIER ! FIN DE TRANSMISSION.")) # NEGATIVE
print(nlp("Je m'attendais à mieux de la part de Franck Dubosc !")) # NEGATIVE
```

## Author

Théophile Blard – :email: theophile.blard@gmail.com

If you use this work (code, model or dataset), please cite as:

> Théophile Blard, French sentiment analysis with BERT, (2020), GitHub repository, <https://github.com/TheophileBlard/french-sentiment-analysis-with-bert>
