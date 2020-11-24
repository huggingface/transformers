---
language: vn
---

# BERT for Vietnamese is trained on more 20 GB news dataset
Bert4news is used for a toolkit Vietnames(segmentation and Named Entity Recognition) at ViNLPtoolkit(https://github.com/bino282/ViNLP)

***************New Mar 11 , 2020 ***************

**[BERT](https://github.com/google-research/bert)** (from Google Research and the Toyota Technological Institute at Chicago) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).

We use word sentencepiece, use basic bert tokenization and same config with bert base with lowercase = False.

You can download trained model:
- [tensorflow](https://drive.google.com/file/d/1X-sRDYf7moS_h61J3L79NkMVGHP-P-k5/view?usp=sharing).
- [pytorch](https://drive.google.com/file/d/11aFSTpYIurn-oI2XpAmcCTccB_AonMOu/view?usp=sharing).


Run demo

``` bash

import torch
from transformers import BertTokenizer,BertModel,AutoTokenizer,AutoModel
tokenizer= AutoTokenizer.from_pretrained("NlpHUST/vibert4news-base-cased")
bert_model = AutoModel.from_pretrained("NlpHUST/vibert4news-base-cased")

line = "Tôi là sinh viên trường Bách Khoa Hà Nội ."
input_id = tokenizer.encode(line,add_special_tokens = True)
att_mask = [int(token_id > 0) for token_id in input_id]
input_ids = torch.tensor([input_ids])
att_masks = torch.tensor([att_mask])
with torch.no_grad():
    features = bert_model(input_ids,att_masks)

print(features)

```

### Contact information
For personal communication related to this project, please contact Nha Nguyen Van (nha282@gmail.com).
