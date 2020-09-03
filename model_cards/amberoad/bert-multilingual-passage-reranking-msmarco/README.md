---
language: multilingual
thumbnail: "https://amberoad.de/images/logo_text.png"
tags:
- msmarco
- multilingual
- passage reranking
license:  Apache-2.0
datasets:
- msmarco
metrics:
- MRR
widget:
- query: "What is a corporation?"
  passage: "A company is incorporated in a specific nation, often within the bounds of a smaller subset of that nation, such as a state or province. The corporation is then governed by the laws of incorporation in that state. A corporation may issue stock, either private or public, or may be classified as a non-stock corporation. If stock is issued, the corporation will usually be governed by its shareholders, either directly or indirectly."
---

# Passage Reranking Multilingual BERT 🔃 🌍



## Model description
**Input:** Supports over 100 Languages. See [List of supported languages](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages) for all available.

**Purpose:** This module takes a search query [1] and a passage [2] and calculates if the passage matches the query. 
It can be used as an improvement for Elasticsearch Results and boosts the relevancy by up to 100%. 

**Architecture:** On top of BERT there is a Densly Connected NN which takes the 768 Dimensional [CLS] Token as input and provides the output ([Arxiv](https://arxiv.org/abs/1901.04085)).

**Output:** Just a single value between between -10 and 10. Better matching query,passage pairs tend to have a higher a score.



## Intended uses & limitations
Both query[1] and passage[2] have to fit in 512 Tokens.
As you normally want to rerank the first dozens of search results keep in mind the inference time of approximately 300 ms/query.

#### How to use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")

model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
```

This Model can be used as a drop-in replacement in the [Nboost Library](https://github.com/koursaros-ai/nboost)
Through this you can directly improve your Elasticsearch Results without any coding. 


## Training data

This model is trained using the [**Microsoft MS Marco Dataset**](https://microsoft.github.io/msmarco/ "Microsoft MS Marco"). This training dataset contains approximately 400M tuples of a query, relevant and non-relevant passages. All datasets used for training and evaluating are listed in this [table](https://github.com/microsoft/MSMARCO-Passage-Ranking#data-information-and-formating). The used dataset for training is called *Train Triples Large*, while the evaluation was made on *Top 1000 Dev*. There are 6,900 queries in total in the development dataset, where each query is mapped to top 1,000 passage retrieved using BM25 from MS MARCO corpus. 

## Training procedure

The training is performed the same way as stated in this [README](https://github.com/nyu-dl/dl4marco-bert "NYU Github"). See their excellent Paper on [Arxiv](https://arxiv.org/abs/1901.04085). 

We changed the BERT Model from an English only to the default BERT Multilingual uncased Model from [Google](https://huggingface.co/bert-base-multilingual-uncased).

Training was done 400 000 Steps. This equaled 12 hours an a TPU V3-8.


## Eval results

We see nearly similar performance than the English only Model in the English [Bing Queries Dataset](http://www.msmarco.org/). Although the training data is English only internal Tests on private data showed a far higher accurancy in German than all other available models.



Fine-tuned Models                                                                   | Dependency                                                                   | Eval Set                                                           | Search Boost<a href='#benchmarks'> | Speed on GPU
----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------- | ----------------------------------
**`amberoad/Multilingual-uncased-MSMARCO`**  (This Model)                                       | <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-blue"/>          |  <a href ='http://www.msmarco.org/'>bing queries</a>               | **+61%** <sub><sup>(0.29 vs 0.18)</sup></sub>         | ~300 ms/query <a href='#footnotes'>
`nboost/pt-tinybert-msmarco`                                          | <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-red"/>          |  <a href ='http://www.msmarco.org/'>bing queries</a>               | **+45%** <sub><sup>(0.26 vs 0.18)</sup></sub>         | ~50ms/query <a href='#footnotes'>
`nboost/pt-bert-base-uncased-msmarco`                                               | <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-red"/>          | <a href ='http://www.msmarco.org/'>bing queries</a>                | **+62%** <sub><sup>(0.29 vs 0.18)</sup></sub>         | ~300 ms/query<a href='#footnotes'>
`nboost/pt-bert-large-msmarco`                                                      | <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-red"/>          | <a href ='http://www.msmarco.org/'>bing queries</a>                | **+77%** <sub><sup>(0.32 vs 0.18)</sup></sub>         | -
`nboost/pt-biobert-base-msmarco`                                                    | <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-red"/>          | <a href ='https://github.com/naver/biobert-pretrained'>biomed</a>  | **+66%** <sub><sup>(0.17 vs 0.10)</sup></sub>         | ~300 ms/query<a href='#footnotes'>

This table is taken from [nboost](https://github.com/koursaros-ai/nboost) and extended by the first line. 



## Contact Infos

![](https://amberoad.de/images/logo_text.png)

Amberoad is a company focussing on Search and Business Intelligence. 
We provide you: 
* Advanced Internal Company Search Engines thorugh NLP
* External Search Egnines: Find Competitors, Customers, Suppliers 

**Get in Contact now to benefit from our Expertise:**

The training and evaluation was performed by [**Philipp Reissel**](https://reissel.eu/) and [**Igli Manaj**](https://github.com/iglimanaj) 

 [![Amberoad](https://i.stack.imgur.com/gVE0j.png) Linkedin](https://de.linkedin.com/company/amberoad) | <svg xmlns="http://www.w3.org/2000/svg" x="0px" y="0px"
width="32" height="32"
viewBox="0 0 172 172"
style=" fill:#000000;"><g fill="none" fill-rule="nonzero" stroke="none" stroke-width="1" stroke-linecap="butt" stroke-linejoin="miter" stroke-miterlimit="10" stroke-dasharray="" stroke-dashoffset="0" font-family="none" font-weight="none" font-size="none" text-anchor="none" style="mix-blend-mode: normal"><path d="M0,172v-172h172v172z" fill="none"></path><g fill="#e67e22"><path d="M37.625,21.5v86h96.75v-86h-5.375zM48.375,32.25h10.75v10.75h-10.75zM69.875,32.25h10.75v10.75h-10.75zM91.375,32.25h32.25v10.75h-32.25zM48.375,53.75h75.25v43h-75.25zM80.625,112.875v17.61572c-1.61558,0.93921 -2.94506,2.2687 -3.88428,3.88428h-49.86572v10.75h49.86572c1.8612,3.20153 5.28744,5.375 9.25928,5.375c3.97183,0 7.39808,-2.17347 9.25928,-5.375h49.86572v-10.75h-49.86572c-0.93921,-1.61558 -2.2687,-2.94506 -3.88428,-3.88428v-17.61572z"></path></g></g></svg>[Homepage](https://de.linkedin.com/company/amberoad) |  [Email](info@amberoad.de)




