## CodeBERT-base-mlm
Pretrained weights for [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155).

### Training Data
The model is trained on the code corpus of [CodeSearchNet](https://github.com/github/CodeSearchNet)

### Training Objective
This model is initialized with Roberta-base and trained with a simple MLM (Masked Language Model) objective.

### Usage
```python
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline

model = RobertaForMaskedLM.from_pretrained('microsoft/codebert-base-mlm')
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')

code_example = "if (x is not None) <mask> (x>1)"
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

outputs = fill_mask(code_example)
print(outputs)
```
Expected results:
```
{'sequence': '<s> if (x is not None) and (x>1)</s>', 'score': 0.6049249172210693, 'token': 8}
{'sequence': '<s> if (x is not None) or (x>1)</s>', 'score': 0.30680200457572937, 'token': 50}
{'sequence': '<s> if (x is not None) if (x>1)</s>', 'score': 0.02133703976869583, 'token': 114}
{'sequence': '<s> if (x is not None) then (x>1)</s>', 'score': 0.018607674166560173, 'token': 172}
{'sequence': '<s> if (x is not None) AND (x>1)</s>', 'score': 0.007619690150022507, 'token': 4248}
```

### Reference
1. [Bimodal CodeBERT trained with MLM+RTD objective](https://huggingface.co/microsoft/codebert-base) (suitable for code search and document generation)
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
