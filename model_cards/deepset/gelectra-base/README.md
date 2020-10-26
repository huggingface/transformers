---
language: de
license: mit
datasets:
- wikipedia
- OPUS
- OpenLegalData
---

# German ELECTRA base

Released, Oct 2020, this is a German ELECTRA language model trained collaboratively by the makers of the original German BERT (aka "bert-base-german-cased") and the dbmdz BERT (aka bert-base-german-dbmdz-cased). In our [paper](https://arxiv.org/pdf/2010.10906.pdf), we outline the steps taken to train our model. Our evaluation suggests that this model is somewhat undertrained. For best performance from a base sized model, we recommend deepset/gbert-base

## Overview  
**Paper:** [here](https://arxiv.org/pdf/2010.10906.pdf)  
**Architecture:** ELECTRA base (discriminator)
**Language:** German  

## Performance  
```
GermEval18 Coarse: 76.02
GermEval18 Fine:   42.22
GermEval14:        86.02
```

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
