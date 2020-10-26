---
language: de
license: mit
datasets:
- wikipedia
- OPUS
- OpenLegalData
---

# German ELECTRA base generator

Released, Oct 2020, this is the generator component of the German ELECTRA language model trained collaboratively by the makers of the original German BERT (aka "bert-base-german-cased") and the dbmdz BERT (aka bert-base-german-dbmdz-cased). In our [paper](https://arxiv.org/pdf/2010.10906.pdf), we outline the steps taken to train our model.

The generator is useful for performing masking experiments. If you are looking for a regular language model for embedding extraction, or downstream tasks like NER, classification or QA, please use deepset/gelectra-base.

## Overview  
**Paper:** [here](https://arxiv.org/pdf/2010.10906.pdf)  
**Architecture:** ELECTRA base (generator)
**Language:** German  

See also:  
deepset/gbert-base
deepset/gbert-large
deepset/gelectra-base
deepset/gelectra-large
deepset/gelectra-base-generator
deepset/gelectra-large-generator

## Authors
Branden Chan: `branden.chan [at] deepset.ai`
Stefan Schweter: `stefan [at] schweter.eu`
Timo Möller: `timo.moeller [at] deepset.ai`

## About us
![deepset logo](https://raw.githubusercontent.com/deepset-ai/FARM/master/docs/img/deepset_logo.png)

We bring NLP to the industry via open source!  
Our focus: Industry specific language models & large scale QA systems.  
  
Some of our work: 
- [German BERT (aka "bert-base-german-cased")](https://deepset.ai/german-bert)
- [FARM](https://github.com/deepset-ai/FARM)
- [Haystack](https://github.com/deepset-ai/haystack/)

Get in touch:
[Twitter](https://twitter.com/deepset_ai) | [LinkedIn](https://www.linkedin.com/company/deepset-ai/) | [Website](https://deepset.ai)
