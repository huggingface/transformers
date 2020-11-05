## xprophetnet-large-wiki100-cased-xglue-ntg
Cross-lingual version [ProphetNet](https://arxiv.org/abs/2001.04063), pretrained on [wiki100 xGLUE dataset](https://arxiv.org/abs/2004.01401) and finetuned on xGLUE cross-lingual News Titles Generation task.  
ProphetNet is a new pre-trained language model for sequence-to-sequence learning with a novel self-supervised objective called future n-gram prediction.  
ProphetNet is able to predict more future tokens with a n-stream decoder. The original implementation is Fairseq version at [github repo](https://github.com/microsoft/ProphetNet).   

xProphetNet is also served as the baseline model for xGLUE cross-lingual natural language generation tasks.  
For xGLUE corss-lingual NLG tasks, xProphetNet is finetuned with English data, but inference with both English and other zero-shot language data.  
### Usage
A quick usage is like: 
```
from transformers import XLMProphetNetTokenizer, XLMProphetNetForConditionalGeneration, ProphetNetConfig

model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/xprophetnet-large-wiki100-cased-xglue-ntg')
tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/xprophetnet-large-wiki100-cased-xglue-ntg')

EN_SENTENCE = "Microsoft Corporation intends to officially end free support for the Windows 7 operating system after January 14, 2020, according to the official portal of the organization. From that day, users of this system will not be able to receive security updates, which could make their computers vulnerable to cyber attacks."
RU_SENTENCE = "орпорация Microsoft намерена официально прекратить бесплатную поддержку операционной системы Windows 7 после 14 января 2020 года, сообщается на официальном портале организации . С указанного дня пользователи этой системы не смогут получать обновления безопасности, из-за чего их компьютеры могут стать уязвимыми к кибератакам."
ZH_SENTENCE = "根据该组织的官方门户网站，微软公司打算在2020年1月14日之后正式终止对Windows 7操作系统的免费支持。从那时起，该系统的用户将无法接收安全更新，这可能会使他们的计算机容易受到网络攻击。"
inputs = tokenizer([EN_SENTENCE, RU_SENTENCE, ZH_SENTENCE], padding=True, max_length=256, return_tensors='pt')

summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

# should give:
# 'Microsoft to end Windows 7 free support after January 14, 2020'
# 'Microsoft намерена прекратить бесплатную поддержку Windows 7 после 14 января 2020 года'
# '微软终止对Windows 7操作系统的免费支持'
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
