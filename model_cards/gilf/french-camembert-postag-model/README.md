---
language: fr
widget:
- text: "Face à un choc inédit, les mesures mises en place par le gouvernement ont permis une protection forte et efficace des ménages"
---

## About

The  *french-camembert-postag-model* is a part of speech tagging model for French that was trained on the *free-french-treebank* dataset available on 
[github](https://github.com/nicolashernandez/free-french-treebank). The base tokenizer and model used for training is *'camembert-base'*.

## Supported Tags

It uses the following tags:

| Tag      |          Category              |  Extra Info |
|----------|:------------------------------:|------------:|
| ADJ      |           adjectif             |             |
| ADJWH    |           adjectif             |             |
| ADV      |           adverbe              |             |
| ADVWH    |           adverbe              |             |
| CC       |  conjonction de coordination   |             |
| CLO      |            pronom              |     obj     |
| CLR      |            pronom              |     refl    |
| CLS      |            pronom              |     suj     |
| CS       |  conjonction de subordination  |             |
| DET      |          déterminant           |             |
| DETWH    |          déterminant           |             |
| ET       |          mot étranger          |             |
| I        |          interjection          |             |
| NC       |          nom commun            |             |
| NPP      |          nom propre            |             |
| P        |          préposition           |             |
| P+D      |   préposition + déterminant    |             |
| PONCT    |      signe de ponctuation      |             |
| PREF     |            préfixe             |             |
| PRO      |        autres pronoms          |             |
| PROREL   |        autres pronoms          |     rel     |
| PROWH    |        autres pronoms          |     int     |
| U        |               ?                |             |
| V        |             verbe              |             |
| VIMP     |        verbe imperatif         |             |
| VINF     |        verbe infinitif         |             |
| VPP      |        participe passé         |             |
| VPR      |        participe présent       |             |
| VS       |        subjonctif              |             |

More information on the tags can be found here:

http://alpage.inria.fr/statgram/frdep/Publications/crabbecandi-taln2008-final.pdf

## Usage

The usage of this model follows the common transformers patterns. Here is a short example of its usage:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("gilf/french-camembert-postag-model")
model = AutoModelForTokenClassification.from_pretrained("gilf/french-camembert-postag-model")

from transformers import pipeline

nlp_token_class = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)

nlp_token_class('Face à un choc inédit, les mesures mises en place par le gouvernement ont permis une protection forte et efficace des ménages')
```

The lines above would display something like this on a Jupyter notebook:

```
[{'entity_group': 'NC', 'score': 0.5760144591331482, 'word': '<s>'},
 {'entity_group': 'U', 'score': 0.9946700930595398, 'word': 'Face'},
 {'entity_group': 'P', 'score': 0.999615490436554, 'word': 'à'},
 {'entity_group': 'DET', 'score': 0.9995906352996826, 'word': 'un'},
 {'entity_group': 'NC', 'score': 0.9995531439781189, 'word': 'choc'},
 {'entity_group': 'ADJ', 'score': 0.999183714389801, 'word': 'inédit'},
 {'entity_group': 'P', 'score': 0.3710663616657257, 'word': ','},
 {'entity_group': 'DET', 'score': 0.9995903968811035, 'word': 'les'},
 {'entity_group': 'NC', 'score': 0.9995649456977844, 'word': 'mesures'},
 {'entity_group': 'VPP', 'score': 0.9988670349121094, 'word': 'mises'},
 {'entity_group': 'P', 'score': 0.9996246099472046, 'word': 'en'},
 {'entity_group': 'NC', 'score': 0.9995329976081848, 'word': 'place'},
 {'entity_group': 'P', 'score': 0.9996233582496643, 'word': 'par'},
 {'entity_group': 'DET', 'score': 0.9995935559272766, 'word': 'le'},
 {'entity_group': 'NC', 'score': 0.9995369911193848, 'word': 'gouvernement'},
 {'entity_group': 'V', 'score': 0.9993771314620972, 'word': 'ont'},
 {'entity_group': 'VPP', 'score': 0.9991101026535034, 'word': 'permis'},
 {'entity_group': 'DET', 'score': 0.9995885491371155, 'word': 'une'},
 {'entity_group': 'NC', 'score': 0.9995636343955994, 'word': 'protection'},
 {'entity_group': 'ADJ', 'score': 0.9991781711578369, 'word': 'forte'},
 {'entity_group': 'CC', 'score': 0.9991298317909241, 'word': 'et'},
 {'entity_group': 'ADJ', 'score': 0.9992275238037109, 'word': 'efficace'},
 {'entity_group': 'P+D', 'score': 0.9993300437927246, 'word': 'des'},
 {'entity_group': 'NC', 'score': 0.8353511393070221, 'word': 'ménages</s>'}]
```
