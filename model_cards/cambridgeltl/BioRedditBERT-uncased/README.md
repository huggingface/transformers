---
language: 
- en
tags:
- BioNLP
- social_media
---

# BioRedditBERT

## Model description
BioRedditBERT is a BERT model initialised from BioBERT (`BioBERT-Base v1.0 + PubMed 200K + PMC 270K`) and further pre-trained on health-related Reddit posts. Please view our paper [COMETA: A Corpus for Medical Entity Linking in the Social Media](https://arxiv.org/pdf/2010.03295.pdf) (EMNLP 2020) for more details.


## Training data

We crawled all threads from 68 health themed subreddits such as `r/AskDocs`, `r/health` and etc. starting from the beginning of 2015 to the end of 2018, obtaining a collection of more than
800K discussions. This collection was then pruned by removing deleted posts, comments from bots or moderators, and so on. In the end, we obtained the training corpus with ca. 300 million tokens and a vocabulary
size of ca. 780,000 words.

## Training procedure
We use the same pre-training script in the original [google-research/bert](https://github.com/google-research/bert) repo. The model is initialised with [`BioBERT-Base v1.0 + PubMed 200K + PMC 270K`](https://github.com/dmis-lab/biobert).
We train with a batch size of 64, a max sequence length of 64, a learning rate of `2e-5` for 100k steps on two GeForce GTX 1080Ti (11 GB) GPUs. Other hyper-parameters are the same as default.


## Eval results
To show the benefit from further pre-training on the social media domain, we demonstrate results on a medical entity linking dataset also in the social media: [AskAPatient](https://zenodo.org/record/55013#.X4ncRmTYpb8) [(Limsopatham and Collier 2016)](https://www.aclweb.org/anthology/P16-1096.pdf). 
We follow the same 10-fold cross-validation procedure for all models and report the average result without fine-tuning. `[CLS]` is used as representations for entity mentions (we also tried average of all tokens but found `[CLS]` generally performs better).

Model   | Accuracy@1 | Accuracy@5
-------|---------|---------
[BERT-base-uncased](https://huggingface.co/bert-base-uncased)  | 38.2 | 43.3
[BioBERT v1.1](https://huggingface.co/dmis-lab/biobert-v1.1) | 41.4 | 51.5
[ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) | 43.9 | 54.3
[BlueBERT](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12.zip) | 41.5 | 48.5
[SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) | 42.3 | 51.9 
[PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) | 42.5 | 49.6
BioRedditBERT | **44.3** | **56.2**


### BibTeX entry and citation info

```bibtex
@inproceedings{basaldella-2020-cometa,
    title = "{COMETA}: A Corpus for Medical Entity Linking in the Social Media",
    author = "Basaldella, Marco  and Liu, Fangyu, and Shareghi, Ehsan, and Collier, Nigel",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020",
    publisher = "Association for Computational Linguistics"
}
```
