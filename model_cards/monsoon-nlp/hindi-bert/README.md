---
language: Hindi
---

# Hindi-BERT (Discriminator)

This is a first run of a Hindi language model trained with Google Research's [ELECTRA](https://github.com/google-research/electra).  **I don't modify ELECTRA until we get into finetuning**

Tokenization and training CoLab: https://colab.research.google.com/drive/1R8TciRSM7BONJRBc9CBZbzOmz39FTLl_

Blog post: https://medium.com/@mapmeld/teaching-hindi-to-electra-b11084baab81

Greatly influenced by: https://huggingface.co/blog/how-to-train

## Corpus

Download: https://drive.google.com/drive/u/1/folders/1WikYHHMI72hjZoCQkLPr45LDV8zm9P7p

The corpus is two files:
- Hindi CommonCrawl deduped by OSCAR https://traces1.inria.fr/oscar/
- latest Hindi Wikipedia ( https://dumps.wikimedia.org/hiwiki/20200420/ ) + WikiExtractor to txt 

Bonus notes:
- Adding English wiki text or parallel corpus could help with cross-lingual tasks and training

## Vocabulary

https://drive.google.com/file/d/1-02Um-8ogD4vjn4t-wD2EwCE-GtBjnzh/view?usp=sharing

Bonus notes:
- Created with HuggingFace Tokenizers; could be longer or shorter, review ELECTRA vocab_size param

## Pretrain TF Records

[build_pretraining_dataset.py](https://github.com/google-research/electra/blob/master/build_pretraining_dataset.py) splits the corpus into training documents

Set the ELECTRA model size and whether to split the corpus by newlines.  This process can take hours on its own.

https://drive.google.com/drive/u/1/folders/1--wBjSH59HSFOVkYi4X-z5bigLnD32R5

Bonus notes:
- I am not sure of the meaning of the corpus newline split (what is the alternative?) and given this corpus, which creates the better training docs

## Training

Structure your files, with data-dir named "trainer" here

```
trainer
- vocab.txt
- pretrain_tfrecords
-- (all .tfrecord... files)
- models
-- modelname
--- checkpoint
--- graph.pbtxt
--- model.*
```

CoLab notebook gives examples of GPU vs. TPU setup

[configure_pretraining.py](https://github.com/google-research/electra/blob/master/configure_pretraining.py)

Model https://drive.google.com/drive/folders/1cwQlWryLE4nlke4OixXA7NK8hzlmUR0c?usp=sharing

## Using this model with Transformers

Sample movie reviews classifier: https://colab.research.google.com/drive/1mSeeSfVSOT7e-dVhPlmSsQRvpn6xC05w
