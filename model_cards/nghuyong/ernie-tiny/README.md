---
language: en
---

# ERNIE-tiny

## Introduction
ERNIE-tiny is the compressed model from [ERNIE 2.0](../ernie-2.0-en) base model through model structure compression and model distillation.
Through compression, the performance of the ERNIE-tiny has only decreased by an average of 2.37% relative to ERNIE 2.0 base, 
but it has increased by 8.35% compared to Google BERT, and the speed has increased by 4.3 times.

More detail: https://github.com/PaddlePaddle/ERNIE/blob/develop/distill/README.md

## Released Model Info

|Model Name|Language|Model Structure|
|:---:|:---:|:---:|
|ernie-tiny| English |Layer:3, Hidden:1024, Heads:16|

This released pytorch model is convert from the official released PaddlePaddle ERNIE model and 
a serial experiments have been made to check the accuracy of the conversion.

- Official PaddlePaddle ERNIE rep: https://github.com/PaddlePaddle/ERNIE
- Pytorch Conversion rep :  https://github.com/nghuyong/ERNIE-Pytorch

## How to use
```Python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-tiny")
model = AutoModel.from_pretrained("nghuyong/ernie-tiny")
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
