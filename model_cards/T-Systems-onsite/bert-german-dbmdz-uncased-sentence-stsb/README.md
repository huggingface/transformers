---
language: de
license: mit
---

# bert-german-dbmdz-uncased-sentence-stsb

## How to use
The usage description above is wrong. Please use this:

Install the `sentence-transformers` package. See here: <https://github.com/UKPLab/sentence-transformers>
```python
from sentence_transformers import models
from sentence_transformers import SentenceTransformer

# load BERT model from Hugging Face
word_embedding_model = models.Transformer(
    'T-Systems-onsite/bert-german-dbmdz-uncased-sentence-stsb')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

# join BERT model and pooling to the sentence transformer
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

## Model description

## Intended uses

## Training data

## Training procedure

## Eval results
