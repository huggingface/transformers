## CodeBERT-base
Pretrained weights for [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155).

### Training Data
The model is trained on bi-modal data (documents & code) of [CodeSearchNet](https://github.com/github/CodeSearchNet)

### Training Objective
This model is initialized with Roberta-base and trained with MLM+RTD objective (cf. the paper).

### Usage
Please see [the official repository](https://github.com/microsoft/CodeBERT) for scripts that support "code search" and "code-to-document generation".

### Reference
1. [CodeBERT trained with Masked LM objective](https://huggingface.co/microsoft/codebert-base-mlm) (suitable for code completion)
2. 🤗 [Hugging Face's CodeBERTa](https://huggingface.co/huggingface/CodeBERTa-small-v1) (small size, 6 layers)

### Citation
```bibtex
@misc{feng2020codebert,
    title={CodeBERT: A Pre-Trained Model for Programming and Natural Languages},
    author={Zhangyin Feng and Daya Guo and Duyu Tang and Nan Duan and Xiaocheng Feng and Ming Gong and Linjun Shou and Bing Qin and Ting Liu and Daxin Jiang and Ming Zhou},
    year={2020},
    eprint={2002.08155},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
