---
language: en
datasets:
- cnn_dailymail
---

## prophetnet-large-uncased-cnndm
Fine-tuned weights(converted from [original fairseq version repo](https://github.com/microsoft/ProphetNet)) for [ProphetNet](https://arxiv.org/abs/2001.04063) on summarization task CNN/DailyMail.  
ProphetNet is a new pre-trained language model for sequence-to-sequence learning with a novel self-supervised objective called future n-gram prediction.  
ProphetNet is able to predict more future tokens with a n-stream decoder. The original implementation is Fairseq version at [github repo](https://github.com/microsoft/ProphetNet).   

### Usage
```
from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration, ProphetNetConfig

model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased-cnndm')
tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased-cnndm')

ARTICLE_TO_SUMMARIZE = "USTC was founded in Beijing by the Chinese Academy of Sciences (CAS) in September 1958. The Director of CAS, Mr. Guo Moruo was appointed the first president of USTC. USTC's founding mission was to develop a high-level science and technology workforce, as deemed critical for development of China's economy, defense, and science and technology education. The establishment was hailed as \"A Major Event in the History of Chinese Education and Science.\" CAS has supported USTC by combining most of its institutes with the departments of the university. USTC is listed in the top 16 national key universities, becoming the youngest national key university.".lower()
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=100, return_tensors='pt')

# Generate Summary
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=512, early_stopping=True)
tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

# should give: 'ustc was founded in beijing by the chinese academy of sciences in 1958. [X_SEP] ustc\'s mission was to develop a high - level science and technology workforce. [X_SEP] the establishment was hailed as " a major event in the history of chinese education and science "'
```

Here, [X_SEP] is used as a special token to seperate sentences.
### Citation
```bibtex
@article{yan2020prophetnet,
  title={Prophetnet: Predicting future n-gram for sequence-to-sequence pre-training},
  author={Yan, Yu and Qi, Weizhen and Gong, Yeyun and Liu, Dayiheng and Duan, Nan and Chen, Jiusheng and Zhang, Ruofei and Zhou, Ming},
  journal={arXiv preprint arXiv:2001.04063},
  year={2020}
}
```
