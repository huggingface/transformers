## About

The  *french-postag-model* is a part of speech tagging model for French that was trained on the *free-french-treebank* dataset available on 
[github](https://github.com/nicolashernandez/free-french-treebank). The base tokenizer and model used for training is *'bert-base-multilingual-cased'*.

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

tokenizer = AutoTokenizer.from_pretrained("gilf/french-postag-model")
model = AutoModelForTokenClassification.from_pretrained("gilf/french-postag-model")

from transformers import pipeline

nlp_token_class = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)

nlp_token_class('Face à un choc inédit, les mesures mises en place par le gouvernement ont permis une protection forte et efficace des ménages')
```

The lines above would display something like this on a Jupyter notebook:

```
[{'entity_group': 'PONCT', 'score': 0.0742340236902237, 'word': '[CLS]'},
 {'entity_group': 'U', 'score': 0.9995399713516235, 'word': 'Face'},
 {'entity_group': 'P', 'score': 0.9999609589576721, 'word': 'à'},
 {'entity_group': 'DET', 'score': 0.9999597072601318, 'word': 'un'},
 {'entity_group': 'NC', 'score': 0.9998948276042938, 'word': 'choc'},
 {'entity_group': 'ADJ', 'score': 0.995318204164505, 'word': 'inédit'},
 {'entity_group': 'PONCT', 'score': 0.9999793171882629, 'word': ','},
 {'entity_group': 'DET', 'score': 0.999964714050293, 'word': 'les'},
 {'entity_group': 'NC', 'score': 0.999936580657959, 'word': 'mesures'},
 {'entity_group': 'VPP', 'score': 0.9995776414871216, 'word': 'mises'},
 {'entity_group': 'P', 'score': 0.99996417760849, 'word': 'en'},
 {'entity_group': 'NC', 'score': 0.999882161617279, 'word': 'place'},
 {'entity_group': 'P', 'score': 0.9999671578407288, 'word': 'par'},
 {'entity_group': 'DET', 'score': 0.9999637603759766, 'word': 'le'},
 {'entity_group': 'NC', 'score': 0.9999350309371948, 'word': 'gouvernement'},
 {'entity_group': 'V', 'score': 0.9999298453330994, 'word': 'ont'},
 {'entity_group': 'VPP', 'score': 0.9998740553855896, 'word': 'permis'},
 {'entity_group': 'DET', 'score': 0.9999625086784363, 'word': 'une'},
 {'entity_group': 'NC', 'score': 0.9999420046806335, 'word': 'protection'},
 {'entity_group': 'ADJ', 'score': 0.9998913407325745, 'word': 'forte'},
 {'entity_group': 'CC', 'score': 0.9998615980148315, 'word': 'et'},
 {'entity_group': 'ADJ', 'score': 0.9998483657836914, 'word': 'efficace'},
 {'entity_group': 'P+D', 'score': 0.9987645149230957, 'word': 'des'},
 {'entity_group': 'NC', 'score': 0.8720395267009735, 'word': 'ménages [SEP]'}]
```
