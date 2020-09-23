---
language: "nl"
thumbnail: "https://github.com/iPieter/RobBERT/raw/master/res/robbert_logo.png"
tags:
- Dutch
- RoBERTa
- RobBERT
license: "MIT"
datasets:
- oscar
- Shuffled Dutch section of the OSCAR corpus (https://oscar-corpus.com/)
---

# RobBERT

## Model description

[RobBERT v2](https://github.com/iPieter/RobBERT) is a Dutch state-of-the-art [RoBERTa](https://arxiv.org/abs/1907.11692)-based language model.

More detailled information can be found in the [RobBERT paper](https://arxiv.org/abs/2001.06286).

## How to use

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
model = RobertaForSequenceClassification.from_pretrained("pdelobelle/robbert-v2-dutch-base")
```

## Performance Evaluation Results

All experiments are described in more detail in our [paper](https://arxiv.org/abs/2001.06286).

### Sentiment analysis
Predicting whether a review is positive or negative using the [Dutch Book Reviews Dataset](https://github.com/benjaminvdb/110kDBRD).

|   Model           | Accuracy [%]             |
|-------------------|--------------------------|
| ULMFiT            | 93.8                     |
| BERTje            | 93.0                     |
| RobBERT v2        | **95.1**                 |

### Die/Dat (coreference resolution)

We measured how well the models are able to do coreference resolution by predicting whether "die" or "dat" should be filled into a sentence.
For this, we used the [EuroParl corpus](https://www.statmt.org/europarl/).

#### Finetuning on whole dataset

|   Model           | Accuracy [%]             |  F1 [%]    |
|-------------------|--------------------------|--------------|
| [Baseline](https://arxiv.org/abs/2001.02943) (LSTM)   |                          | 75.03        |
| mBERT             | 98.285                   | 98.033       |
| BERTje            | 98.268                   | 98.014       |
| RobBERT v2        | **99.232**               | **99.121**   |

#### Finetuning on 10K examples

We also measured the performance using only 10K training examples.
This experiment clearly illustrates that RobBERT outperforms other models when there is little data available.

|   Model           | Accuracy [%]             |  F1 [%]      |
|-------------------|--------------------------|--------------|
| mBERT             | 92.157                   | 90.898       |
| BERTje            | 93.096                   | 91.279       |
| RobBERT v2        | **97.816**               | **97.514**   |

#### Using zero-shot word masking task

Since BERT models are pre-trained using the word masking task, we can use this to predict whether "die" or "dat" is more likely.
This experiment shows that RobBERT has internalised more information about Dutch than other models.

|   Model           | Accuracy [%]             |
|-------------------|--------------------------|
| ZeroR             | 66.70                    |
| mBERT             | 90.21                    |
| BERTje            | 94.94                    |
| RobBERT v2        | **98.75**                |

### Part-of-Speech Tagging.

Using the [Lassy UD dataset](https://universaldependencies.org/treebanks/nl_lassysmall/index.html).


|   Model           | Accuracy [%]             |
|-------------------|--------------------------|
| Frog              | 91.7                     |
| mBERT             | **96.5**                 |
| BERTje            | 96.3                     |
| RobBERT v2        | 96.4                     |

Interestingly, we found that when dealing with **small data sets**, RobBERT v2 **significantly outperforms** other models.

<p align="center"> 
    <img src="https://github.com/iPieter/RobBERT/blob/master/res/robbert_pos_accuracy.png" alt="RobBERT's performance on smaller datasets">
 </p>

### Named Entity Recognition

Using the [CoNLL 2002 evaluation script](https://www.clips.uantwerpen.be/conll2002/ner/).


|   Model           | Accuracy [%]             |
|-------------------|--------------------------|
| Frog              | 57.31                    |
| mBERT             | **90.94**                |
| BERT-NL           | 89.7                     |
| BERTje            | 88.3                     |
| RobBERT v2        | 89.08                    |


## Training procedure

We pre-trained RobBERT using the RoBERTa training regime.
We pre-trained our model on the Dutch section of the [OSCAR corpus](https://oscar-corpus.com/), a large multilingual corpus which was obtained by language classification in the Common Crawl corpus.
This Dutch corpus is 39GB large, with 6.6 billion words spread over 126 million lines of text, where each line could contain multiple sentences, thus using more data than concurrently developed Dutch BERT models.


RobBERT shares its architecture with [RoBERTa's base model](https://github.com/pytorch/fairseq/tree/master/examples/roberta), which itself is a replication and improvement over BERT.
Like BERT, it's architecture consists of 12 self-attention layers with 12 heads with 117M trainable parameters.
One difference with the original BERT model is due to the different pre-training task specified by RoBERTa, using only the MLM task and not the NSP task.
During pre-training, it thus only predicts which words are masked in certain positions of given sentences.
The training process uses the Adam optimizer with polynomial decay of the learning rate l_r=10^-6 and a ramp-up period of 1000 iterations, with hyperparameters beta_1=0.9
and RoBERTa's default beta_2=0.98.
Additionally, a weight decay of 0.1 and a small dropout of 0.1 helps prevent the model from overfitting. 


RobBERT was trained on a computing cluster with 4 Nvidia P100 GPUs per node, where the number of nodes was dynamically adjusted while keeping a fixed batch size of 8192 sentences.
At most 20 nodes were used (i.e. 80 GPUs), and the median was 5 nodes.
By using gradient accumulation, the batch size could be set independently of the number of GPUs available, in order to maximally utilize the cluster.
Using the [Fairseq library](https://github.com/pytorch/fairseq/tree/master/examples/roberta), the model trained for two epochs, which equals over 16k batches in total, which took about three days on the computing cluster.
In between training jobs on the computing cluster, 2 Nvidia 1080 Ti's also covered some parameter updates for RobBERT v2.


## Limitations and bias

In the [RobBERT paper](https://arxiv.org/abs/2001.06286), we also investigated potential sources of bias in RobBERT.

We found that the zeroshot model estimates the probability of *hij* (he) to be higher than *zij* (she) for most occupations in bleached template sentences, regardless of their actual job gender ratio in reality.

<p align="center"> 
    <img src="https://github.com/iPieter/RobBERT/blob/master/res/gender_diff.png" alt="RobBERT's performance on smaller datasets">
 </p>

By augmenting the DBRB Dutch Book sentiment analysis dataset with the stated gender of the author of the review, we found that highly positive reviews written by women were generally more accurately detected by RobBERT as being positive than those written by men.

<p align="center"> 
    <img src="https://github.com/iPieter/RobBERT/blob/master/res/dbrd.png" alt="RobBERT's performance on smaller datasets">
 </p>



## BibTeX entry and citation info

```bibtex
@misc{delobelle2020robbert,
    title={RobBERT: a Dutch RoBERTa-based Language Model},
    author={Pieter Delobelle and Thomas Winters and Bettina Berendt},
    year={2020},
    eprint={2001.06286},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
