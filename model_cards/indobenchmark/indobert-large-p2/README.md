---
language: id
tags:
- indobert
- indobenchmark
- indonlu
license: mit
inference: false
datasets:
- Indo4B
---

# IndoBERT Large Model (phase2 - uncased)

[IndoBERT](https://arxiv.org/abs/2009.05387) is a state-of-the-art language model for Indonesian based on the BERT model. The pretrained model is trained using a masked language modeling (MLM) objective and next sentence prediction (NSP) objective. 

## All Pre-trained Models

| Model                          | #params                        | Arch. | Training data                     |
|--------------------------------|--------------------------------|-------|-----------------------------------|
| `indobenchmark/indobert-base-p1` | 124.5M   | Base  | Indo4B (23.43 GB of text)            |
| `indobenchmark/indobert-base-p2` | 124.5M   | Base  | Indo4B (23.43 GB of text)            |
| `indobenchmark/indobert-large-p1` | 335.2M   | Large  | Indo4B (23.43 GB of text)            |
| `indobenchmark/indobert-large-p2` | 335.2M   | Large  | Indo4B (23.43 GB of text)            |
| `indobenchmark/indobert-lite-base-p1` | 11.7M   | Base  | Indo4B (23.43 GB of text)            |
| `indobenchmark/indobert-lite-base-p2` | 11.7M   | Base  | Indo4B (23.43 GB of text)            |
| `indobenchmark/indobert-lite-large-p1` | 17.7M   | Large  | Indo4B (23.43 GB of text)            |
| `indobenchmark/indobert-lite-large-p2` | 17.7M   | Large  | Indo4B (23.43 GB of text)            |

## How to use

### Load model and tokenizer
```python
from transformers import BertTokenizer, AutoModel
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-large-p2")
model = AutoModel.from_pretrained("indobenchmark/indobert-large-p2")
```

### Extract contextual representation
```python
x = torch.LongTensor(tokenizer.encode('aku adalah anak [MASK]')).view(1,-1)
print(x, model(x)[0].sum())
```

## Authors 

<b>IndoBERT</b> was trained and evaluated by Bryan Wilie\*, Karissa Vincentio\*, Genta Indra Winata\*, Samuel Cahyawijaya\*, Xiaohong Li, Zhi Yuan Lim, Sidik Soleman, Rahmad Mahendra, Pascale Fung, Syafri Bahar, Ayu Purwarianti.


## Citation
If you use our work, please cite:

```bibtex
@inproceedings{wilie2020indonlu,
  title={IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
  author={Bryan Wilie and Karissa Vincentio and Genta Indra Winata and Samuel Cahyawijaya and X. Li and Zhi Yuan Lim and S. Soleman and R. Mahendra and Pascale Fung and Syafri Bahar and A. Purwarianti},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  year={2020}
}
```
