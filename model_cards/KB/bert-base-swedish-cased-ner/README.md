---
language: sv
---

# Swedish BERT Models

The National Library of Sweden / KBLab releases three pretrained language models based on BERT and ALBERT. The models are trained on approximately 15-20GB of text (200M sentences, 3000M tokens) from various sources (books, news, government publications, swedish wikipedia and internet forums) aiming to provide a representative BERT model for Swedish text. A more complete description will be published later on.

The following three models are currently available:

- **bert-base-swedish-cased** (*v1*) - A BERT trained with the same hyperparameters as first published by Google.
- **bert-base-swedish-cased-ner** (*experimental*) - a BERT fine-tuned for NER using SUC 3.0.
- **albert-base-swedish-cased-alpha** (*alpha*) - A first attempt at an ALBERT for Swedish.

All models are cased and trained with whole word masking.

## Files

| **name**                        | **files** |
|---------------------------------|-----------|
| bert-base-swedish-cased         | [config](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased/config.json), [vocab](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased/vocab.txt), [pytorch_model.bin](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased/pytorch_model.bin) |
| bert-base-swedish-cased-ner     | [config](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased-ner/config.json), [vocab](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased-ner/vocab.txt) [pytorch_model.bin](https://s3.amazonaws.com/models.huggingface.co/bert/KB/bert-base-swedish-cased-ner/pytorch_model.bin) |
| albert-base-swedish-cased-alpha | [config](https://s3.amazonaws.com/models.huggingface.co/bert/KB/albert-base-swedish-cased-alpha/config.json), [sentencepiece model](https://s3.amazonaws.com/models.huggingface.co/bert/KB/albert-base-swedish-cased-alpha/spiece.model), [pytorch_model.bin](https://s3.amazonaws.com/models.huggingface.co/bert/KB/albert-base-swedish-cased-alpha/pytorch_model.bin) |

TensorFlow model weights will be released soon.

## Usage requirements / installation instructions

The examples below require Huggingface Transformers 2.4.1 and Pytorch 1.3.1 or greater. For Transformers<2.4.0 the tokenizer must be instantiated manually and the `do_lower_case` flag parameter set to `False` and `keep_accents` to `True` (for ALBERT).

To create an environment where the examples can be run, run the following in an terminal on your OS of choice.

```
# git clone https://github.com/Kungbib/swedish-bert-models
# cd swedish-bert-models
# python3 -m venv venv
# source venv/bin/activate
# pip install --upgrade pip
# pip install -r requirements.txt
```

### BERT Base Swedish

A standard BERT base for Swedish trained on a variety of sources. Vocabulary size is ~50k. Using Huggingface Transformers the model can be loaded in Python as follows:

```python
from transformers import AutoModel,AutoTokenizer

tok = AutoTokenizer.from_pretrained('KB/bert-base-swedish-cased')
model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')
```


### BERT base fine-tuned for Swedish NER

This model is fine-tuned on the SUC 3.0 dataset. Using the Huggingface pipeline the model can be easily instantiated. For Transformer<2.4.1 it seems the tokenizer must be loaded separately to disable lower-casing of input strings:

```python
from transformers import pipeline

nlp = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner')

nlp('Idag släpper KB tre språkmodeller.')
```

Running the Python code above should produce in something like the result below. Entity types used are `TME` for time, `PRS` for personal names, `LOC` for locations, `EVN` for events and `ORG` for organisations. These labels are subject to change.

```python
[ { 'word': 'Idag', 'score': 0.9998126029968262, 'entity': 'TME' },
  { 'word': 'KB',   'score': 0.9814832210540771, 'entity': 'ORG' } ]
```

The BERT tokenizer often splits words into multiple tokens, with the subparts starting with `##`, for example the string `Engelbert kör Volvo till Herrängens fotbollsklubb` gets tokenized as `Engel ##bert kör Volvo till Herr ##ängens fotbolls ##klubb`. To glue parts back together one can use something like this:

```python
text = 'Engelbert tar Volvon till Tele2 Arena för att titta på Djurgården IF ' +\
       'som spelar fotboll i VM klockan två på kvällen.'

l = []
for token in nlp(text):
    if token['word'].startswith('##'):
        l[-1]['word'] += token['word'][2:]
    else:
        l += [ token ]

print(l)
```

Which should result in the following (though less cleanly formatted):

```python
[ { 'word': 'Engelbert',     'score': 0.99..., 'entity': 'PRS'},
  { 'word': 'Volvon',        'score': 0.99..., 'entity': 'OBJ'},
  { 'word': 'Tele2',         'score': 0.99..., 'entity': 'LOC'},
  { 'word': 'Arena',         'score': 0.99..., 'entity': 'LOC'},
  { 'word': 'Djurgården',    'score': 0.99..., 'entity': 'ORG'},
  { 'word': 'IF',            'score': 0.99..., 'entity': 'ORG'},
  { 'word': 'VM',            'score': 0.99..., 'entity': 'EVN'},
  { 'word': 'klockan',       'score': 0.99..., 'entity': 'TME'},
  { 'word': 'två',           'score': 0.99..., 'entity': 'TME'},
  { 'word': 'på',            'score': 0.99..., 'entity': 'TME'},
  { 'word': 'kvällen',       'score': 0.54..., 'entity': 'TME'} ]
```

### ALBERT base

The easiest way to do this is, again, using Huggingface Transformers:

```python
from transformers import AutoModel,AutoTokenizer

tok = AutoTokenizer.from_pretrained('KB/albert-base-swedish-cased-alpha'),
model = AutoModel.from_pretrained('KB/albert-base-swedish-cased-alpha')
```

## Acknowledgements ❤️

- Resources from Stockholms University, Umeå University and Swedish Language Bank at Gothenburg University were used when fine-tuning BERT for NER.
- Model pretraining was made partly in-house at the KBLab and partly (for material without active copyright) with the support of Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
- Models are hosted on S3 by Huggingface 🤗

