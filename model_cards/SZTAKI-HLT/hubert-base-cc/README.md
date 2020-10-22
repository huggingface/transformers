---
language: hu
license: apache-2.0
datasets:
- common_crawl
- wikipedia
---

# huBERT base model (cased)

## Model description

Cased BERT model for Hungarian, trained on the (filtered, deduplicated) Hungarian subset of the Common Crawl and a snapshot of the Hungarian Wikipedia.

## Intended uses & limitations

The model can be used as any other (cased) BERT model. It has been tested on the chunking and
named entity recognition tasks and set a new state-of-the-art on the former.

## Training

Details of the training data and procedure can be found in the PhD thesis linked below. (With the caveat that it only contains preliminary results
based on the Wikipedia subcorpus. Evaluation of the full model will appear in a future paper.)

## Eval results

When fine-tuned (via `BertForTokenClassification`) on chunking and NER, the model outperforms multilingual BERT, achieves state-of-the-art results on the
former task and comes within 0.5% F1 to the SotA on the latter. The exact scores are

| NER | Minimal NP | Maximal NP |
|-----|------------|------------|
| 97.62% | **97.14%** | **96.97%** |

### BibTeX entry and citation info

The training corpus, parameters and the evaluation methods are discussed in the
[following PhD thesis](https://hlt.bme.hu/en/publ/nemeskey_2020):

```bibtex
@PhDThesis{ Nemeskey:2020,
  author = {Nemeskey, Dávid Márk},
  title  = {Natural Language Processing Methods for Language Modeling},
  year   = {2020},
  school = {E\"otv\"os Lor\'and University}
}
```
