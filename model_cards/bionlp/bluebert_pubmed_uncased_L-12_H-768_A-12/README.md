---
language: 
- en
tags:
- bluebert
license: 
- PUBLIC DOMAIN NOTICE
datasets:
- pubmed

---

# BlueBert-Base, Uncased, PubMed

## Model description

A BERT model pre-trained on PubMed abstracts

## Intended uses & limitations

#### How to use

Please see https://github.com/ncbi-nlp/bluebert

## Training data

We provide [preprocessed PubMed texts](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/pubmed_uncased_sentence_nltk.txt.tar.gz) that were used to pre-train the BlueBERT models. 
The corpus contains ~4000M words extracted from the [PubMed ASCII code version](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PubMed/). 

Pre-trained model: https://huggingface.co/bert-base-uncased

## Training procedure

*  lowercasing the text
*  removing speical chars `\x00`-`\x7F`
*  tokenizing the text using the [NLTK Treebank tokenizer](https://www.nltk.org/_modules/nltk/tokenize/treebank.html)

Below is a code snippet for more details.

```python
value = value.lower()
value = re.sub(r'[\r\n]+', ' ', value)
value = re.sub(r'[^\x00-\x7F]+', ' ', value)

tokenized = TreebankWordTokenizer().tokenize(value)
sentence = ' '.join(tokenized)
sentence = re.sub(r"\s's\b", "'s", sentence)
```

### BibTeX entry and citation info

```bibtex
@InProceedings{peng2019transfer,
  author    = {Yifan Peng and Shankai Yan and Zhiyong Lu},
  title     = {Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets},
  booktitle = {Proceedings of the 2019 Workshop on Biomedical Natural Language Processing (BioNLP 2019)},
  year      = {2019},
  pages     = {58--65},
}
```
