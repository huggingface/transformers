---
language: en
datasets:
- c4
tags:
- summarization
- translation

license: apache-2.0
inference: false
---

## Disclaimer

Due do it's immense size, `t5-11b` requires some special treatment. 
First, `t5-11b` should be loaded with flag `use_cdn` set to `False` as follows:

```python
t5 = transformers.T5ForConditionalGeneration.from_pretrained('t5-11b', use_cdn = False)
```

Secondly, a single GPU will most likely not have enough memory to even load the model into memory as the weights alone amount to over 40 GB.
Model parallelism has to be used here to overcome this problem as is explained in this [PR](https://github.com/huggingface/transformers/pull/3578).

## [Google's T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) 

Pretraining Dataset: [C4](https://huggingface.co/datasets/c4)

Other Community Checkpoints: [here](https://huggingface.co/models?search=t5)

Paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)

Authors: *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu* 

## Abstract

Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code.

![model image](https://camo.githubusercontent.com/623b4dea0b653f2ad3f36c71ebfe749a677ac0a1/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f6d61782f343030362f312a44304a31674e51663876727255704b657944387750412e706e67)

