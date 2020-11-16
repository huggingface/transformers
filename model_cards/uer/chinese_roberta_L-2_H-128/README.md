---
language: zh
datasets: 
- CLUECorpus
---

# Chinese RoBERTa Miniatures

## Model description

This is the set of 24 Chinese RoBERTa models pre-trained by [UER-py](https://www.aclweb.org/anthology/D19-3041.pdf).

You can download the 24 Chinese RoBERTa miniatures either from the [UER-py Github page](https://github.com/dbiir/UER-py/), or via HuggingFace from the links below:

|   |H=128|H=256|H=512|H=768|
|---|:---:|:---:|:---:|:---:|
| **L=2**  |[**2/128 (BERT-Tiny)**][2_128]|[2/256]|[2/512]|[2/768]|
| **L=4**  |[4/128]|[**4/256 (BERT-Mini)**]|[**4/512 (BERT-Small)**]|[4/768]|
| **L=6**  |[6/128]|[6/256]|[6/512]|[6/768]|
| **L=8**  |[8/128]|[8/256]|[**8/512 (BERT-Medium)**]|[8/768]|
| **L=10** |[10/128]|[10/256]|[10/512]|[10/768]|
| **L=12** |[12/128]|[12/256]|[12/512]|[**12/768 (BERT-Base)**]|

## Training data

CLUECorpus2020 and CLUECorpusSmall are used as training corpus.

## Training procedure

Training details can be found in [UER-py](https://github.com/dbiir/UER-py/).

### BibTeX entry and citation info

```
@article{zhao2019uer,
  title={UER: An Open-Source Toolkit for Pre-training Models},
  author={Zhao, Zhe and Chen, Hui and Zhang, Jinbin and Zhao, Xin and Liu, Tao and Lu, Wei and Chen, Xi and Deng, Haotang and Ju, Qi and Du, Xiaoyong},
  journal={EMNLP-IJCNLP 2019},
  pages={241},
  year={2019}
}
```

[2_128]: https://huggingface.co/uer/chinese_roberta_L-2_H-128
