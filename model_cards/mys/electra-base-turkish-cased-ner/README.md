---
language: tr
---

## What is this

A NER model for Turkish with 48 categories trained on the dataset [Shrinked TWNERTC Turkish NER Data](https://www.kaggle.com/behcetsenturk/shrinked-twnertc-turkish-ner-data-by-kuzgunlar) by Behçet Şentürk, which is itself a filtered and cleaned version of the following automatically labeled dataset:

> Sahin, H. Bahadir; Eren, Mustafa Tolga; Tirkaz, Caglar; Sonmez, Ozan; Yildiz, Eray (2017), “English/Turkish Wikipedia Named-Entity Recognition and Text Categorization Dataset”, Mendeley Data, v1 http://dx.doi.org/10.17632/cdcztymf4k.1

## Backbone model

The backbone model is [electra-base-turkish-cased-discriminator](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator), and I finetuned it for token classification.

I'm continuing to figure out if it is possible to improve accuracy with this dataset, but it is already usable for non-critic applications. You can reach out to me on [Twitter](https://twitter.com/myusufsarigoz) for discussions and issues. 
I will also release a notebook to finetune NER models with Shrinked TWNERTC as well as sample inference code to demonstrate what's possible with this model.
