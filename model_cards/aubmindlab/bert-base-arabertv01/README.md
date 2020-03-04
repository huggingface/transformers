---
language: arabic
---

# AraBERT : Pre-training BERT for Arabic Language Understanding

**AraBERT** is an Arabic pretrained lanaguage model based on [Google's BERT architechture](https://github.com/google-research/bert). AraBERT uses the same BERT-Base config.

There are two version off the model AraBERTv0.1 and AraBERTv1, with the difference being that AraBERTv1 uses pre-segmented text where prefixes and suffixes were splitted using the [Farasa Segmenter](http://alt.qcri.org/farasa/segmenter.html).

The model was trained on ~70M sentences or ~23GB of Arabic text with ~3B words. The training corpora are a collection of publically available large scale raw arabic text ([Arabic Wikidumps](https://archive.org/details/arwiki-20190201), [The 1.5B words Arabic Corpus](https://www.semanticscholar.org/paper/1.5-billion-words-Arabic-Corpus-El-Khair/f3eeef4afb81223df96575adadf808fe7fe440b4), [The OSIAN Corpus](https://www.aclweb.org/anthology/W19-4619), Assafir news articles, and 4 other manually crawled news websites (Al-Akhbar, Annahar, AL-Ahram, AL-Wafd) from [the Wayback Machine](http://web.archive.org/))

We evalaute both AraBERT models on different downstream tasks and compare it to [mBERT]((https://github.com/google-research/bert/blob/master/multilingual.md)), and other state of the art models (*To the extent of our knowledge*). The Tasks were Sentiment Analysis on 6 different datasets ([HARD](https://github.com/elnagara/HARD-Arabic-Dataset), [ASTD-Balanced](https://www.aclweb.org/anthology/D15-1299), [ArsenTD-Lev](https://staff.aub.edu.lb/~we07/Publications/ArSentD-LEV_Sentiment_Corpus.pdf), [LABR](https://github.com/mohamedadaly/LABR), [ArSaS](http://lrec-conf.org/workshops/lrec2018/W30/pdf/22_W30.pdf)), Named Entity Recognition with the [ANERcorp](http://curtis.ml.cmu.edu/w/courses/index.php/ANERcorp), and Arabic Question Answering on [Arabic-SQuAD and ARCD](https://github.com/husseinmozannar/SOQAL)

## Results (Acc.)
Task | prev. SOTA | mBERT | AraBERTv0.1 | AraBERTv1
---|:---:|:---:|:---:|:---:
HARD |95.7 [ElJundi et.al.](https://www.aclweb.org/anthology/W19-4608/)|95.7|96.2|96.1
ASTD |86.5 [ElJundi et.al.](https://www.aclweb.org/anthology/W19-4608/)| 80.1|92.2|92.6
ArsenTD-Lev|52.4 [ElJundi et.al.](https://www.aclweb.org/anthology/W19-4608/)|51|58.9|59.4
AJGT|93 [Dahou et.al.](https://dl.acm.org/doi/fullHtml/10.1145/3314941)| 83.6|94.1|93.8
LABR|87.5 [Dahou et.al.](https://dl.acm.org/doi/fullHtml/10.1145/3314941)|83|85.9|86.7
ANERcorp|81.7 (BiLSTM-CRF)|78.4|84.2|81.9
ARCD|mBERT|EM:34.2 F1: 61.3|EM:30.1 F1:61.2|EM:30.6 F1: 62.7

*We would be extremly thankful if everyone can contibute to the Results table by adding more scores on different datasets*

## How to use

You can easily use AraBERT since it is almost fully compatible with existing codebases (You can use this repo instead of the official BERT one, the only difference is in the ```tokenization.py``` file where we modify the _is_punctuation function to make it compatible with the "+" symbol and the "[" and "]" characters)

To use HuggingFace's Transformer repository you only need to provide a lost of token that forces the model to not split them, also make sure that the text is pre-segmented:

```python
from transformers import AutoTokenizer
from preprocess_arabert import never_split_tokens

arabert_tokenizer = AutoTokenizer.from_pretrained(
    "aubmindlab/bert-base-arabert",
    do_lower_case=False,
    do_basic_tokenize=True,
    never_split=never_split_tokens)
arabert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabert")

arabert_tokenizer.tokenize("و+ لن نبالغ إذا قل +نا إن هاتف أو كمبيوتر ال+ مكتب في زمن +نا هذا ضروري")

>>> ['و+', 'لن', 'نبال', '##غ', 'إذا', 'قل', '+نا', 'إن', 'هاتف', 'أو', 'كمبيوتر', 'ال+', 'مكتب', 'في', 'زمن', '+نا', 'هذا', 'ضروري']
```

**AraBERTv0.1 is compatible with all existing libraries, since it needs no pre-segmentation.**
```python
from transformers import AutoTokenizer
from preprocess_arabert import never_split_tokens

arabert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv01",do_lower_case=False)
arabert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv01")

arabert_tokenizer.tokenize("ولن نبالغ إذا قلنا إن هاتف أو كمبيوتر المكتب في زمننا هذا ضروري")

>>> ['ولن', 'ن', '##بالغ', 'إذا', 'قلنا', 'إن', 'هاتف', 'أو', 'كمبيوتر', 'المكتب', 'في', 'زمن', '##ن', '##ا', 'هذا', 'ضروري']
```


The ```araBERT_(initial_Demo_TF)_.ipynb``` Notebook is a small demo using the AJGT dataset using TensorFlow (GPU and TPU compatible).

## Model Weights and Vocab Download
Models | AraBERTv0.1 | AraBERTv1
---|:---:|:---:
TensorFlow|[Drive Link](https://drive.google.com/open?id=1-kVmTUZZ4DP2rzeHNjTPkY8OjnQCpomO) | [Drive Link](https://drive.google.com/open?id=1-d7-9ljKgDJP5mx73uBtio-TuUZCqZnt)
PyTorch| [Drive_Link](https://drive.google.com/open?id=1-_3te42mQCPD8SxwZ3l-VBL7yaJH-IOv)| [Drive_Link](https://drive.google.com/open?id=1-69s6Pxqbi63HOQ1M9wTcr-Ovc6PWLLo)

**You can find the PyTorch models in HuggingFace's Transformer Library under the ```aubmindlab``` username**

## If you used this model please cite us as:
```
@misc{antoun2020arabert,
    title={AraBERT: Transformer-based Model for Arabic Language Understanding},
    author={Wissam Antoun and Fady Baly and Hazem Hajj},
    year={2020},
    eprint={2003.00104},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
## Acknowledgments 
Thanks to TensorFlow Research Cloud (TFRC) for the free access to Cloud TPUs, couldn't have done it without this program, and to the [AUB MIND Lab](https://sites.aub.edu.lb/mindlab/) Members for the continous support. Also thanks to [Yakshof](https://www.yakshof.com/#/) and Assafir for data and storage access.

## Contacts
**Wissam Antoun**: [Linkedin](https://www.linkedin.com/in/giulio-ravasio-3a81a9110/) | [Twitter](https://twitter.com/wissam_antoun) | [Github](https://github.com/WissamAntoun) | <wfa07@mail.aub.edu> | <wissam.antoun@gmail.com>

**Fady Baly**: [Linkedin](https://www.linkedin.com/in/fadybaly/) | [Twitter](https://twitter.com/BalyFady) | [Github](https://github.com/fadybaly) | <fgb06@mail.aub.edu> | <baly.fady@gmail.com>

***We are looking for sponsors to train BERT-Large and other Transformer models, the sponsor only needs to cover to data storage and compute cost of the generating the pretraining data***
