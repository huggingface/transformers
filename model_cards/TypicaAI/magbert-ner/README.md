---
language: fr
widget:
- text: "Je m'appelle Hicham et je vis a Fès"
---

# MagBERT-NER: a state-of-the-art NER model for Moroccan French language (Maghreb)

## Introduction

[MagBERT-NER] is a state-of-the-art NER model for Moroccan French language (Maghreb). The MagBERT-NER model was fine-tuned for NER Task based the language model for French Camembert (based on the RoBERTa architecture).

For further information or requests, please go to [Typica.AI Website](https://typicasoft.io/)

## How to use MagBERT-NER with HuggingFace

##### Load MagBERT-NER and its sub-word tokenizer :
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("TypicaAI/magbert-ner")
model = AutoModelForTokenClassification.from_pretrained("TypicaAI/magbert-ner")


##### Process text sample (from wikipedia about the current Prime Minister of Morocco) Using NER pipeline  

from transformers import pipeline

nlp = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)
nlp("Saad Dine El Otmani, né le 16 janvier 1956 à Inezgane, est un homme d'État marocain, chef du gouvernement du Maroc depuis le 5 avril 2017")


#[{'entity_group': 'I-PERSON',
#  'score': 0.8941445276141167,
#  'word': 'Saad Dine El Otmani'},
# {'entity_group': 'B-DATE',
#  'score': 0.5967703461647034,
#  'word': '16 janvier 1956'},
# {'entity_group': 'B-GPE', 'score': 0.7160899192094803, 'word': 'Inezgane'},
# {'entity_group': 'B-NORP', 'score': 0.7971733212471008, 'word': 'marocain'},
# {'entity_group': 'B-GPE', 'score': 0.8921478390693665, 'word': 'Maroc'},
# {'entity_group': 'B-DATE',
#  'score': 0.5760444005330404,
#  'word': '5 avril 2017'}]

```

```


## Authors 

MagBert-NER was trained and evaluated by Hicham Assoudi, Ph.D.


