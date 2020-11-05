---
language: de
license: mit
tags:
- sentence_embedding
- search
- pytorch 
- xlm-roberta 
- roberta
- xlm-r-distilroberta-base-paraphrase-v1
- paraphrase
datasets:
- STSbenchmark
metrics:
- Spearman’s rank correlation
- cosine similarity
---

# German RoBERTa for Sentence Embeddings V2
**The new [T-Systems-onsite/cross-en-de-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer) model is slightly better for German language. It is also the current best model for English language and works cross-lingually. Please consider using that model.**

This model is intended to [compute sentence (text embeddings)](https://www.sbert.net/docs/usage/computing_sentence_embeddings.html) for German text. These embeddings can then be compared with [cosine-similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to find sentences with a similar semantic meaning. For example this can be useful for [semantic textual similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html), [semantic search](https://www.sbert.net/docs/usage/semantic_search.html), or [paraphrase mining](https://www.sbert.net/docs/usage/paraphrase_mining.html). To do this you have to use the [Sentence Transformers Python framework](https://github.com/UKPLab/sentence-transformers).

> Sentence-BERT (SBERT) is a  modification  of  the  pretrained BERT network that use siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity. This reduces the effort for finding the most similar pair from 65hours with BERT / RoBERTa to about 5 seconds with SBERT, while maintaining the accuracy from BERT.

Source: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

This model is fine-tuned from [Philip May](https://eniak.de/) and open-sourced by [T-Systems-onsite](https://www.t-systems-onsite.de/). Special thanks to [Nils Reimers](https://www.nils-reimers.de/) for your awesome open-source work, the Sentence Transformers, the models and your help on GitHub.

## How to use
**The usage description above - provided by Hugging Face - is wrong for sentence embeddings! Please use this:**

To use this model install the `sentence-transformers` package (see here: <https://github.com/UKPLab/sentence-transformers>).

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('T-Systems-onsite/german-roberta-sentence-transformer-v2')
```

For details of usage and examples see here:
- [Computing Sentence Embeddings](https://www.sbert.net/docs/usage/computing_sentence_embeddings.html)
- [Semantic Textual Similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
- [Paraphrase Mining](https://www.sbert.net/docs/usage/paraphrase_mining.html)
- [Semantic Search](https://www.sbert.net/docs/usage/semantic_search.html)
- [Cross-Encoders](https://www.sbert.net/docs/usage/cross-encoder.html)
- [Examples on GitHub](https://github.com/UKPLab/sentence-transformers/tree/master/examples)

## Training
The base model is [xlm-roberta-base](https://huggingface.co/xlm-roberta-base). This model has been further trained by [Nils Reimers](https://www.nils-reimers.de/) on a large scale paraphrase dataset for 50+ languages. [Nils Reimers](https://www.nils-reimers.de/) about this [on GitHub](https://github.com/UKPLab/sentence-transformers/issues/509#issuecomment-712243280):

>A paper is upcoming for the paraphrase models.
>
>These models were trained on various datasets with Millions of examples for paraphrases, mainly derived from Wikipedia edit logs, paraphrases mined from Wikipedia and SimpleWiki, paraphrases from news reports, AllNLI-entailment pairs with in-batch-negative loss etc.
>
>In internal tests, they perform much better than the NLI+STSb models as they have see more and broader type of training data. NLI+STSb has the issue that they are rather narrow in their domain and do not contain any domain specific words / sentences (like from chemistry, computer science, math etc.). The paraphrase models has seen plenty of sentences from various domains.
>
>More details with the setup, all the datasets, and a wider evaluation will follow soon.

The resulting model called `xlm-r-distilroberta-base-paraphrase-v1` has been released here: <https://github.com/UKPLab/sentence-transformers/releases/tag/v0.3.8>

Building on this cross language model we fine-tuned it for German language on the [deepl.com](https://www.deepl.com/translator) dataset of our [German STSbenchmark dataset](https://github.com/t-systems-on-site-services-gmbh/german-STSbenchmark).

We did an automatic hyperparameter search for 102 trials with [Optuna](https://github.com/optuna/optuna). Using 10-fold crossvalidation on the deepl.com test and dev dataset we found the following best hyperparameters:
- batch_size = 15
- num_epochs = 4
- lr = 2.2995320905210864e-05
- eps = 1.8979875906303792e-06
- weight_decay = 0.003314045812507563
- warmup_steps_proportion = 0.46141685205829014

The final model was trained with these hyperparameters on the combination of `sts_de_train.csv` and `sts_de_dev.csv`. The `sts_de_test.csv` was left for testing.

# Evaluation
The evaluation has been done on the test set of our [German STSbenchmark dataset](https://github.com/t-systems-on-site-services-gmbh/german-STSbenchmark). The code is available on [Colab](https://colab.research.google.com/drive/1aCWOqDQx953kEnQ5k4Qn7uiixokocOHv?usp=sharing). As the metric for evaluation we use the Spearman’s rank correlation between the  cosine-similarity of the sentence embeddings and STSbenchmark labels.

| Model Name                           | Spearman rank correlation<br/>(German)           |
|--------------------------------------|-------------------------------------|
| xlm-r-distilroberta-base-paraphrase-v1                        | 0.8079     |
| xlm-r-100langs-bert-base-nli-stsb-mean-tokens                 | 0.8194     |
| xlm-r-bert-base-nli-stsb-mean-tokens                          | 0.8194     |
| **T-Systems-onsite/<br/>german-roberta-sentence-transformer-v2**   | **0.8529** |
| **[T-Systems-onsite/<br/>cross-en-de-roberta-sentence-transformer](https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer)** | **0.8550** |
