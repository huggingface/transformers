## xprophetnet-large-wiki100-cased-xglue-ntg
Cross-lingual version [ProphetNet](https://arxiv.org/abs/2001.04063), pretrained on [wiki100 xGLUE dataset](https://arxiv.org/abs/2004.01401) and finetuned on xGLUE cross-lingual Question Generation task.  
ProphetNet is a new pre-trained language model for sequence-to-sequence learning with a novel self-supervised objective called future n-gram prediction.  
ProphetNet is able to predict more future tokens with a n-stream decoder. The original implementation is Fairseq version at [github repo](https://github.com/microsoft/ProphetNet).   

xProphetNet is also served as the baseline model for xGLUE cross-lingual natural language generation tasks.  
For xGLUE corss-lingual NLG tasks, xProphetNet is finetuned with English data, but inference with both English and other zero-shot language data.  
### Usage
A quick usage is like: 
```
from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration, ProphetNetConfig

model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/xprophetnet-large-wiki100-cased-xglue-qg')
tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/xprophetnet-large-wiki100-cased-xglue-qg')

EN_SENTENCE = "Google left China in 2010"
ZH_SENTENCE = "Google在2010年离开中国"
inputs = tokenizer([EN_SENTENCE, ZH_SENTENCE], padding=True, max_length=256, return_tensors='pt')

summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
print([tokenizer.decode(g) for g in summary_ids])  
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
