---
language: en
---

## prophetnet-large-uncased
Pretrained weights for [ProphetNet](https://arxiv.org/abs/2001.04063).  
ProphetNet is a new pre-trained language model for sequence-to-sequence learning with a novel self-supervised objective called future n-gram prediction.  
ProphetNet is able to predict more future tokens with a n-stream decoder. The original implementation is Fairseq version at [github repo](https://github.com/microsoft/ProphetNet).   

### Usage

This pre-trained model can be fine-tuned on *sequence-to-sequence* tasks. The model could *e.g.* be trained on headline generation as follows:

```python 
from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer

model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased")
tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")

input_str = "the us state department said wednesday it had received no formal word from bolivia that it was expelling the us ambassador there but said the charges made against him are `` baseless ."
target_str = "us rejects charges against its ambassador in bolivia"

input_ids = tokenizer(input_str, return_tensors="pt").input_ids
labels = tokenizer(target_str, return_tensors="pt").input_ids

loss = model(input_ids, labels=labels, return_dict=True).loss
```

### Citation
```bibtex
@article{yan2020prophetnet,
  title={Prophetnet: Predicting future n-gram for sequence-to-sequence pre-training},
  author={Yan, Yu and Qi, Weizhen and Gong, Yeyun and Liu, Dayiheng and Duan, Nan and Chen, Jiusheng and Zhang, Ruofei and Zhou, Ming},
  journal={arXiv preprint arXiv:2001.04063},
  year={2020}
}
```
