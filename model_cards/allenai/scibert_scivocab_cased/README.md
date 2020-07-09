# SciBERT

This is the pretrained model presented in [SciBERT: A Pretrained Language Model for Scientific Text](https://www.aclweb.org/anthology/D19-1371/), which is a BERT model trained on scientific text.

The training corpus was papers taken from [Semantic Scholar](https://www.semanticscholar.org). Corpus size is 1.14M papers, 3.1B tokens. We use the full text of the papers in training, not just abstracts.

SciBERT has its own wordpiece vocabulary (scivocab) that's built to best match the training corpus. We trained cased and uncased versions. 

Available models include:
* `scibert_scivocab_cased`
* `scibert_scivocab_uncased`


The original repo can be found [here](https://github.com/allenai/scibert).

If using these models, please cite the following paper:
```
@inproceedings{beltagy-etal-2019-scibert,
    title = "SciBERT: A Pretrained Language Model for Scientific Text",
    author = "Beltagy, Iz  and Lo, Kyle  and Cohan, Arman",
    booktitle = "EMNLP",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1371"
}
```
