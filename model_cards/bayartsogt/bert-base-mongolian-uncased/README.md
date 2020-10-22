---
language: "mn"
tags:
- bert
- mongolian
- uncased
---

# BERT-BASE-MONGOLIAN-UNCASED
[Link to Official Mongolian-BERT repo](https://github.com/tugstugi/mongolian-bert)

## Model description
This repository contains pre-trained Mongolian [BERT](https://arxiv.org/abs/1810.04805) models trained by [tugstugi](https://github.com/tugstugi), [enod](https://github.com/enod) and [sharavsambuu](https://github.com/sharavsambuu).
Special thanks to [nabar](https://github.com/nabar) who provided 5x TPUs.

This repository is based on the following open source projects: [google-research/bert](https://github.com/google-research/bert/),
[huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and [yoheikikuta/bert-japanese](https://github.com/yoheikikuta/bert-japanese).

#### How to use

```python
from transformers import pipeline, AlbertTokenizer, BertForMaskedLM

tokenizer = AlbertTokenizer.from_pretrained('bayartsogt/bert-base-mongolian-uncased')
model = BertForMaskedLM.from_pretrained('bayartsogt/bert-base-mongolian-uncased')

## declare task ##
pipe = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)

## example ##
input_  = 'Миний [MASK] хоол идэх нь тун чухал.'

output_ = pipe(input_)
for i in range(len(output_)):
    print(output_[i])

```


## Training data
Mongolian Wikipedia and the 700 million word Mongolian news data set  [[Pretraining Procedure](https://github.com/tugstugi/mongolian-bert#pre-training)]

### BibTeX entry and citation info

```bibtex
@misc{mongolian-bert,
  author = {Tuguldur, Erdene-Ochir and Gunchinish, Sharavsambuu and Bataa, Enkhbold},
  title = {BERT Pretrained Models on Mongolian Datasets},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tugstugi/mongolian-bert/}}
}
```
