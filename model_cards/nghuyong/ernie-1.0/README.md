---
language: zh
---

# ERNIE-1.0

## Introduction

ERNIE (Enhanced Representation through kNowledge IntEgration) is proposed by Baidu in 2019,
which is designed to learn language representation enhanced by knowledge masking strategies i.e. entity-level masking and phrase-level masking. 
Experimental results show that ERNIE achieve state-of-the-art results on five Chinese natural language processing tasks including natural language inference, 
semantic similarity, named entity recognition, sentiment analysis and question answering. 

More detail: https://arxiv.org/abs/1904.09223

## Released Model Info

|Model Name|Language|Model Structure|
|:---:|:---:|:---:|
|ernie-1.0| Chinese |Layer:12, Hidden:768, Heads:12|

This released pytorch model is converted from the officially released PaddlePaddle ERNIE model and 
a series of experiments have been conducted to check the accuracy of the conversion.

- Official PaddlePaddle ERNIE repo: https://github.com/PaddlePaddle/ERNIE
- Pytorch Conversion repo:  https://github.com/nghuyong/ERNIE-Pytorch

## How to use
```Python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
model = AutoModel.from_pretrained("nghuyong/ernie-1.0")
```

## Citation

```bibtex
@article{sun2019ernie,
  title={Ernie: Enhanced representation through knowledge integration},
  author={Sun, Yu and Wang, Shuohuan and Li, Yukun and Feng, Shikun and Chen, Xuyi and Zhang, Han and Tian, Xin and Zhu, Danxiang and Tian, Hao and Wu, Hua},
  journal={arXiv preprint arXiv:1904.09223},
  year={2019}
}
```
