---
language: "mn"
tags:
- mongolian
- cased
---

# BERT-BASE-MONGOLIAN-CASED
[Link to Official Mongolian-BERT repo](https://github.com/tugstugi/mongolian-bert)

## Model description
This repository contains pre-trained Mongolian [BERT](https://arxiv.org/abs/1810.04805) models trained by [tugstugi](https://github.com/tugstugi), [enod](https://github.com/enod) and [sharavsambuu](https://github.com/sharavsambuu).
Special thanks to [nabar](https://github.com/nabar) who provided 5x TPUs.

This repository is based on the following open source projects: [google-research/bert](https://github.com/google-research/bert/),
[huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and [yoheikikuta/bert-japanese](https://github.com/yoheikikuta/bert-japanese).

#### How to use

```python
from transformers import pipeline, AlbertTokenizer, BertForMaskedLM

tokenizer = AlbertTokenizer.from_pretrained('bayartsogt/bert-base-mongolian-cased')
model = BertForMaskedLM.from_pretrained('bayartsogt/bert-base-mongolian-cased')

## declare task ##
pipe = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)

## example ##
input_  = 'Миний [MASK] хоол идэх нь тун чухал.'

output_ = pipe(input_)
for i in range(len(output_)):
    print(output_[i])
    
## Output ##
# {'sequence': '[CLS] Миний хувьд хоол идэх нь тун чухал.[SEP]', 'score': 0.8734784722328186, 'token': 95, 'token_str': '▁хувьд'}
# {'sequence': '[CLS] Миний бодлоор хоол идэх нь тун чухал.[SEP]', 'score': 0.09788835793733597, 'token': 6320, 'token_str': '▁бодлоор'}
# {'sequence': '[CLS] Миний хүү хоол идэх нь тун чухал.[SEP]', 'score': 0.0027510314248502254, 'token': 590, 'token_str': '▁хүү'}
# {'sequence': '[CLS] Миний бие хоол идэх нь тун чухал.[SEP]', 'score': 0.0014857524074614048, 'token': 267, 'token_str': '▁бие'}
# {'sequence': '[CLS] Миний охин хоол идэх нь тун чухал.[SEP]', 'score': 0.0013575413031503558, 'token': 1116, 'token_str': '▁охин'}

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
