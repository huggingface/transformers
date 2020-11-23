---
language: id
tags:
- indobert
- indolem
license: mit
inference: false
datasets:
- 220M words (IndoWiki, IndoWC, News)
---

## About

[IndoBERT](https://arxiv.org/pdf/2011.00677.pdf) is the Indonesian version of BERT model. We train the model using over 220M words, aggregated from three main sources: 
* Indonesian Wikipedia (74M words)
* news articles from Kompas, Tempo (Tala et al., 2003), and Liputan6 (55M words in total)
* an Indonesian Web Corpus (Medved and Suchomel, 2017) (90M words).

We trained the model for 2.4M steps (180 epochs) with the final perplexity over the development set being <b>3.97</b> (similar to English BERT-base).

This <b>IndoBERT</b> was used to examine IndoLEM - an Indonesian benchmark that comprises of seven tasks for the Indonesian language, spanning morpho-syntax, semantics, and discourse. 

| Task | Metric | Bi-LSTM | mBERT | MalayBERT | IndoBERT |
| ---- | ---- | ---- | ---- | ---- | ---- |
| POS Tagging | Acc | 95.4 | <b>96.8</b> | <b>96.8</b> | <b>96.8</b> |
| NER UGM | F1| 70.9 | 71.6 | 73.2 | <b>74.9</b> |
| NER UI | F1 | 82.2 | 82.2 | 87.4 | <b>90.1</b> |
| Dep. Parsing (UD-Indo-GSD) | UAS/LAS | 85.25/80.35 | 86.85/81.78 | 86.99/81.87 | <b>87.12<b/>/<b>82.32</b> |
| Dep. Parsing (UD-Indo-PUD) | UAS/LAS | 84.04/79.01 | <b>90.58</b>/<b>85.44</b> | 88.91/83.56 | 89.23/83.95 |
| Sentiment Analysis | F1 | 71.62 | 76.58 | 82.02 | <b>84.13</b> |
| Summarization | R1/R2/RL | 67.96/61.65/67.24 | 68.40/61.66/67.67 | 68.44/61.38/67.71 | <b>69.93</b>/<b>62.86</b>/<b>69.21</b> |
| Next Tweet Prediction | Acc | 73.6 | 92.4 | 93.1 | <b>93.7</b> |
| Tweet Ordering | Spearman corr. | 0.45 | 0.53 | 0.51 | <b>0.59</b> |

The paper is published at the 28th COLING 2020. Please refer to https://indolem.github.io for more details about the benchmarks.

## How to use

### Load model and tokenizer (tested with transformers==3.5.1)
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
```

## Citation
If you use our work, please cite:

```bibtex
@inproceedings{koto2020indolem,
  title={IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP},
  author={Fajri Koto and Afshin Rahimi and Jey Han Lau and Timothy Baldwin},
  booktitle={Proceedings of the 28th COLING},
  year={2020}
}
```
