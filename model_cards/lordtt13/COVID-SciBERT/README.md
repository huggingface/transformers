---
language: en
inference: false
---

## COVID-SciBERT: A small language modelling expansion of SciBERT, a BERT model trained on scientific text.

### Details of SciBERT

The **SciBERT** model was presented in [SciBERT: A Pretrained Language Model for Scientific Text](https://arxiv.org/abs/1903.10676) by *Iz Beltagy, Kyle Lo, Arman Cohan* and here is the abstract:

Obtaining large-scale annotated data for NLP tasks in the scientific domain is challenging and expensive. We release SciBERT, a pretrained language model based on BERT (Devlin et al., 2018) to address the lack of high-quality, large-scale labeled scientific data. SciBERT leverages unsupervised pretraining on a large multi-domain corpus of scientific publications to improve performance on downstream scientific NLP tasks. We evaluate on a suite of tasks including sequence tagging, sentence classification and dependency parsing, with datasets from a variety of scientific domains. We demonstrate statistically significant improvements over BERT and achieve new state-of-the-art results on several of these tasks.

### Details of the downstream task (Language Modeling) - Dataset 📚

There are actually two datasets that have been used here:

- The original SciBERT model is trained on papers from the corpus of [semanticscholar.org](semanticscholar.org). Corpus size is 1.14M papers, 3.1B tokens. They used the full text of the papers in training, not just abstracts. SciBERT has its own vocabulary (scivocab) that's built to best match the training corpus.

- The expansion is done using the papers present in the [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge). Only the abstracts have been used and vocabulary was pruned and added to the existing scivocab. In response to the COVID-19 pandemic, the White House and a coalition of leading research groups have prepared the COVID-19 Open Research Dataset (CORD-19). CORD-19 is a resource of over 200,000 scholarly articles, including over 100,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses. This freely available dataset is provided to the global research community to apply recent advances in natural language processing and other AI techniques to generate new insights in support of the ongoing fight against this infectious disease. There is a growing urgency for these approaches because of the rapid acceleration in new coronavirus literature, making it difficult for the medical research community to keep up.

### Model training

The training script is present [here](https://github.com/lordtt13/word-embeddings/blob/master/COVID-19%20Research%20Data/COVID-SciBERT.ipynb).

### Pipelining the Model

```python
import transformers

model = transformers.AutoModelWithLMHead.from_pretrained('lordtt13/COVID-SciBERT')

tokenizer = transformers.AutoTokenizer.from_pretrained('lordtt13/COVID-SciBERT')

nlp_fill = transformers.pipeline('fill-mask', model = model, tokenizer = tokenizer)
nlp_fill('Coronavirus or COVID-19 can be prevented by a' + nlp_fill.tokenizer.mask_token)

# Output:
# [{'sequence': '[CLS] coronavirus or covid - 19 can be prevented by a combination [SEP]',
#   'score': 0.1719885915517807,
#   'token': 2702},
#  {'sequence': '[CLS] coronavirus or covid - 19 can be prevented by a simple [SEP]',
#   'score': 0.054218728095293045,
#   'token': 2177},
#  {'sequence': '[CLS] coronavirus or covid - 19 can be prevented by a novel [SEP]',
#   'score': 0.043364267796278,
#   'token': 3045},
#  {'sequence': '[CLS] coronavirus or covid - 19 can be prevented by a high [SEP]',
#   'score': 0.03732519596815109,
#   'token': 597},
#  {'sequence': '[CLS] coronavirus or covid - 19 can be prevented by a vaccine [SEP]',
#   'score': 0.021863549947738647,
#   'token': 7039}]
```

> Created by [Tanmay Thakur](https://github.com/lordtt13) | [LinkedIn](https://www.linkedin.com/in/tanmay-thakur-6bb5a9154/)

> PS: Still looking for more resources to expand my expansion!
