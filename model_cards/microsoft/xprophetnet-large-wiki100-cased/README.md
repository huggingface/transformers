---
language: multilingual
---

## xprophetnet-large-wiki100-cased
Cross-lingual version [ProphetNet](https://arxiv.org/abs/2001.04063), pretrained on [wiki100 xGLUE dataset](https://arxiv.org/abs/2004.01401).  
ProphetNet is a new pre-trained language model for sequence-to-sequence learning with a novel self-supervised objective called future n-gram prediction.  
ProphetNet is able to predict more future tokens with a n-stream decoder. The original implementation is Fairseq version at [github repo](https://github.com/microsoft/ProphetNet).   

xProphetNet is also served as the baseline model for xGLUE cross-lingual natural language generation tasks.  
For xGLUE corss-lingual NLG tasks, xProphetNet is finetuned with English data, but inference with both English and other zero-shot language data. 

### Usage

This pre-trained model can be fine-tuned on *sequence-to-sequence* tasks. The model could *e.g.* be trained on English headline generation as follows:

```python 
from transformers import XLMProphetNetForConditionalGeneration, XLMProphetNetTokenizer

model = XLMProphetNetForConditionalGeneration.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
tokenizer = XLMProphetNetTokenizer.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")

input_str = "the us state department said wednesday it had received no formal word from bolivia that it was expelling the us ambassador there but said the charges made against him are `` baseless ."
target_str = "us rejects charges against its ambassador in bolivia"

input_ids = tokenizer(input_str, return_tensors="pt").input_ids
labels = tokenizer(target_str, return_tensors="pt").input_ids

loss = model(input_ids, labels=labels, return_dict=True).loss
```

Note that since this model is a multi-lingual model it can be fine-tuned on all kinds of other languages.

### Citation
```bibtex
@article{yan2020prophetnet,
  title={Prophetnet: Predicting future n-gram for sequence-to-sequence pre-training},
  author={Yan, Yu and Qi, Weizhen and Gong, Yeyun and Liu, Dayiheng and Duan, Nan and Chen, Jiusheng and Zhang, Ruofei and Zhou, Ming},
  journal={arXiv preprint arXiv:2001.04063},
  year={2020}
}
```
