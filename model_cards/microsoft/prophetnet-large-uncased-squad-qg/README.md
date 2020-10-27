---
language: en
datasets:
- squad
---

## 
prophetnet-large-uncased-squad-qg
Fine-tuned weights(converted from [original fairseq version repo](https://github.com/microsoft/ProphetNet)) for [ProphetNet](https://arxiv.org/abs/2001.04063) on question generation 
SQuAD 1.1.  
ProphetNet is a new pre-trained language model for sequence-to-sequence learning with a novel self-supervised objective called future n-gram prediction.  
ProphetNet is able to predict more future tokens with a n-stream decoder. The original implementation is Fairseq version at [github repo](https://github.com/microsoft/ProphetNet).   

### Usage
```
from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration, ProphetNetConfig

model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased-squad-qg')
tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased-squad-qg')

FACT_TO_GENERATE_QUESTION_FROM = ""Bill Gates [SEP] Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975."

inputs = tokenizer([FACT_TO_GENERATE_QUESTION_FROM], return_tensors='pt')

# Generate Summary
question_ids = model.generate(inputs['input_ids'], num_beams=5, early_stopping=True)
tokenizer.batch_decode(question_ids, skip_special_tokens=True)

# should give: 'along with paul allen, who founded microsoft?'
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
