# LIMIT-BERT

Code and model for the *EMNLP 2020 Findings* paper: 

[LIMIT-BERT: Linguistic Informed Multi-task BERT](https://arxiv.org/abs/1910.14296)) 

## Contents

1. [Requirements](#Requirements)
2. [Training](#Training)

## Requirements

* Python 3.6 or higher.
* Cython 0.25.2 or any compatible version.
* [PyTorch](http://pytorch.org/) 1.0.0+. 
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation. 
* [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) PyTorch 1.0.0+ or any compatible version.

#### Pre-trained Models (PyTorch)
The following pre-trained models are available for download from Google Drive:
* [`LIMIT-BERT`](https://drive.google.com/open?id=1fm0cK2A91iLG3lCpwowCCQSALnWS2X4i): 
  PyTorch version, same setting with BERT-Large-WWM，loading model with [pytorch-transformers](https://github.com/huggingface/pytorch-transformers).

## How to use

```
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("cooelf/limitbert")
model = AutoModel.from_pretrained("cooelf/limitbert")
```

Please see our original repo for the training scripts.

https://github.com/cooelf/LIMIT-BERT

## Training

To train LIMIT-BERT, simply run:
```
sh run_limitbert.sh
```
### Evaluation Instructions

To test after setting model path:
```
sh test_bert.sh
```

## Citation

```
@article{zhou2019limit,
  title={{LIMIT-BERT}: Linguistic informed multi-task {BERT}},
  author={Zhou, Junru and Zhang, Zhuosheng and Zhao, Hai},
  journal={arXiv preprint arXiv:1910.14296},
  year={2019}
}
```