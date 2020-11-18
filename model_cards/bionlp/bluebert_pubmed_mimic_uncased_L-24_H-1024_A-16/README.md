---
language: 
- en
tags:
- bert
- bluebert
license: 
- PUBLIC DOMAIN NOTICE
datasets:
- PubMed
- MIMIC-III

---

# BlueBert-Base, Uncased, PubMed and MIMIC-III

## Model description

A BERT model pre-trained on PubMed abstracts and clinical notes ([MIMIC-III](https://mimic.physionet.org/)).

## Intended uses & limitations

#### How to use

Please see https://github.com/ncbi-nlp/bluebert

## Training data

We provide [preprocessed PubMed texts](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/pubmed_uncased_sentence_nltk.txt.tar.gz) that were used to pre-train the BlueBERT models. 
The corpus contains ~4000M words extracted from the [PubMed ASCII code version](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PubMed/). 

Pre-trained model: https://huggingface.co/bert-large-uncased

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

### Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of
Medicine and Clinical Center. This work was supported by the National Library of Medicine of the National Institutes of Health under award number 4R00LM013001-01.

We are also grateful to the authors of BERT and ELMo to make the data and codes publicly available.

We would like to thank Dr Sun Kim for processing the PubMed texts.

### Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NCBI. The information produced
on this website is not intended for direct diagnostic use or medical decision-making without review and oversight
by a clinical professional. Individuals should not change their health behavior solely on the basis of information
produced on this website. NIH does not independently verify the validity or utility of the information produced
by this tool. If you have questions about the information produced on this website, please see a health care
professional. More information about NCBI's disclaimer policy is available.
