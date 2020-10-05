---
language: dv
---

# dv-wave

This is a second attempt at a Dhivehi language model trained with
Google Research's [ELECTRA](https://github.com/google-research/electra).

Tokenization and pre-training CoLab: https://colab.research.google.com/drive/1ZJ3tU9MwyWj6UtQ-8G7QJKTn-hG1uQ9v?usp=sharing

Using SimpleTransformers to classify news https://colab.research.google.com/drive/1KnyQxRNWG_yVwms_x9MUAqFQVeMecTV7?usp=sharing

V1: similar performance to mBERT on news classification task after finetuning for 3 epochs (52%)

V2: fixed tokenizers ```do_lower_case=False``` and ```strip_accents=False``` to preserve vowel signs of Dhivehi
  dv-wave: 89% to mBERT: 52%

## Corpus

Trained on @Sofwath's 307MB corpus of Dhivehi text: https://github.com/Sofwath/DhivehiDatasets - this repo also contains the news classification task CSV

[OSCAR](https://oscar-corpus.com/) was considered but has not been added to pretraining; as of
this writing their web crawl has 126MB of Dhivehi text (79MB deduped).

## Vocabulary

Included as vocab.txt in the upload - vocab_size is 29874
