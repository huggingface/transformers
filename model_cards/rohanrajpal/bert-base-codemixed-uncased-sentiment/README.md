---
language:
- hi
- en
tags:
- hi
- en
- codemix
datasets:
- SAIL 2017
---

# Model name

## Model description

I took a bert-base-multilingual-cased model from huggingface and finetuned it on SAIL 2017 dataset.  

## Intended uses & limitations

#### How to use

```python
# You can include sample code which will be formatted
#Coming soon!
```

#### Limitations and bias

Provide examples of latent issues and potential remediations.

## Training data

I trained on the SAIL 2017 dataset [link](http://amitavadas.com/SAIL/Data/SAIL_2017.zip) on this [pretrained model](https://huggingface.co/bert-base-multilingual-cased).


## Training procedure

No preprocessing.

## Eval results

### BibTeX entry and citation info

```bibtex
@inproceedings{khanuja-etal-2020-gluecos,
    title = "{GLUEC}o{S}: An Evaluation Benchmark for Code-Switched {NLP}",
    author = "Khanuja, Simran  and
      Dandapat, Sandipan  and
      Srinivasan, Anirudh  and
      Sitaram, Sunayana  and
      Choudhury, Monojit",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.329",
    pages = "3575--3585"
}
```
