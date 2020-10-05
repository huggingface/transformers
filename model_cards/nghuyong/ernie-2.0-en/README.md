---
language: en
---

# ERNIE-2.0

## Introduction

ERNIE 2.0 is a continual pre-training framework proposed by Baidu in 2019, 
which builds and learns incrementally pre-training tasks through constant multi-task learning. 
Experimental results demonstrate that ERNIE 2.0 outperforms BERT and XLNet on 16 tasks including English tasks on GLUE benchmarks and several common tasks in Chinese. 

More detail: https://arxiv.org/abs/1907.12412

## Released Model Info

|Model Name|Language|Model Structure|
|:---:|:---:|:---:|
|ernie-2.0-en| English |Layer:12, Hidden:768, Heads:12|

This released pytorch model is converted from the officially released PaddlePaddle ERNIE model and 
a series of experiments have been conducted to check the accuracy of the conversion.

- Official PaddlePaddle ERNIE repo: https://github.com/PaddlePaddle/ERNIE
- Pytorch Conversion repo:  https://github.com/nghuyong/ERNIE-Pytorch

## How to use
```Python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
model = AutoModel.from_pretrained("nghuyong/ernie-2.0-en")
```

## Citation

```bibtex
@article{sun2019ernie20,
  title={ERNIE 2.0: A Continual Pre-training Framework for Language Understanding},
  author={Sun, Yu and Wang, Shuohuan and Li, Yukun and Feng, Shikun and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:1907.12412},
  year={2019} 
}
```
