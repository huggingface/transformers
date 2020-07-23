---
language: de
license: mit
---

# bert-german-dbmdz-uncased-sentence-stsb

## How to use
**The usage description above - provided by Hugging Face - is wrong! Please use this:**

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

# join BERT model and pooling to get the sentence transformer
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

## Model description
This is a German [sentence embedding](https://github.com/UKPLab/sentence-transformers) trained on the [German STSbenchmark Dataset](https://github.com/t-systems-on-site-services-gmbh/german-STSbenchmark). It was trained from [Philip May](https://eniak.de/) and open-sourced by [T-Systems-onsite](https://www.t-systems-onsite.de/).The base language model is the [dbmdz/bert-base-german-uncased](https://huggingface.co/dbmdz/bert-base-german-uncased) from [Bayerische Staatsbibliothek ](https://huggingface.co/dbmdz).

## Intended uses
> Sentence-BERT (SBERT) is a  modification  of  the  pretrained BERT network that use siamese and triplet network structures to derive semantically mean-ingful sentence embeddings that can be compared using cosine-similarity. This reduces the effort for finding the most similar pair from 65hours with BERT / RoBERTa to about 5 seconds with SBERT, while maintaining the accuracy from BERT.

Source: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

## Training procedure
We did an automatic hyperprameter optimization with [Optuna](https://github.com/optuna/optuna) and found the following hyperprameters:
- batch_size = 5
- num_epochs = 11
- lr = 2.637549780860126e-05
- eps = 5.0696075038683e-06
- weight_decay = 0.02817210102940054
- warmup_steps = 27.342745941760147 % of total steps

The final model was trained on the combination of all three datasets: `sts_de_dev.csv`, `sts_de_test.csv` and `sts_de_train.csv`
