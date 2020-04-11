# CovidBERT-NLI

This is the model **CovidBERT** trained by DeepSet on AllenAI's [CORD19 Dataset](https://pages.semanticscholar.org/coronavirus-research) of scientific articles about coronaviruses.

The model uses the original BERT wordpiece vocabulary and was subsequently fine-tuned on the [SNLI](https://nlp.stanford.edu/projects/snli/) and the [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) datasets using the [`sentence-transformers` library](https://github.com/UKPLab/sentence-transformers/) to produce universal sentence embeddings [1] using the **average pooling strategy** and a **softmax loss**.

Parameter details for the original training on CORD-19 are available on [DeepSet's MLFlow](https://public-mlflow.deepset.ai/#/experiments/2/runs/ba27d00c30044ef6a33b1d307b4a6cba)

**Base model**: `deepset/covid_bert_base` from HuggingFace's `AutoModel`.

**Training time**: ~6 hours on the NVIDIA Tesla P100 GPU provided in Kaggle Notebooks.

**Parameters**:

| Parameter        | Value |
|------------------|-------|
| Batch size       | 64    |
| Training steps   | 23000 |
| Warmup steps     | 1450  |
| Lowercasing      | True  |
| Max. Seq. Length | 128   |

**Performances**: The performance was evaluated on the test portion of the [STS dataset](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) using Spearman rank correlation and compared to the performances of similar models obtained with the same procedure to verify its performances.

| Model                         | Score       |
|-------------------------------|-------------|
| `covidbert-nli` (this)        | 67.52       |
| `gsarti/biobert-nli`          | 73.40       |
| `gsarti/scibert-nli`          | 74.50       |
| `bert-base-nli-mean-tokens`[2]| 77.12       |

An example usage for similarity-based scientific paper retrieval is provided in the [Covid-19 Semantic Browser](https://github.com/gsarti/covid-papers-browser) repository.

**References:**

[1] A. Conneau et al., [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://www.aclweb.org/anthology/D17-1070/)

[2] N. Reimers et I. Gurevych, [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://www.aclweb.org/anthology/D19-1410/)
