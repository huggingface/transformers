---
language: 
- "zh"
thumbnail: "https://user-images.githubusercontent.com/9592150/97142000-cad08e00-179a-11eb-88df-aff9221482d8.png"
tags:
- "chinese"
- "classical chinese"
- "literary chinese"
- "ancient chinese"
- "bert"
- "pytorch"
license: "apache-2.0"
pipeline_tag: "fill-mask"
widget:
- text: "[MASK]太元中，武陵人捕鱼为业。"
- text: "问征夫以前路，恨晨光之[MASK]微。"
- text: "浔阳江头夜送客，枫叶[MASK]花秋瑟瑟。"
---

# GuwenBERT

## Model description
![GuwenBERT](https://user-images.githubusercontent.com/9592150/97142000-cad08e00-179a-11eb-88df-aff9221482d8.png)

This is a RoBERTa model pre-trained on Classical Chinese. You can fine-tune GuwenBERT for downstream tasks, such as sentence breaking, punctuation, named entity recognition, and so on.

For more information about RoBERTa, take a look at the RoBERTa's offical repo.

## How to use

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-large")

model = AutoModel.from_pretrained("ethanyt/guwenbert-large")
```

## Training data

The training data is daizhige dataset (殆知阁古代文献) which is contains of 15,694 books in Classical Chinese, covering Buddhism, Confucianism, Medicine, History, Zi, Yi, Yizang, Shizang, Taoism, and Jizang. 
76% of them are punctuated.
The total number of characters is 1.7B (1,743,337,673).
All traditional Characters are converted to simplified characters.
The vocabulary is constructed from this data set and the size is 23,292.

## Training procedure

The models are initialized with `hfl/chinese-roberta-wwm-ext-large` and then pre-trained with a 2-step strategy.
In the first step, the model learns MLM with only word embeddings updated during training, until convergence. In the second step, all parameters are updated during training.

The models are trained on 4 V100 GPUs for 120K steps (20K for step#1, 100K for step#2) with a batch size of 2,048 and a sequence length of 512. The optimizer used is Adam with a learning rate of 1e-4, adam-betas of (0.9,0.98), adam-eps of 1e-6, a weight decay of 0.01, learning rate warmup for 5K steps, and linear decay of learning rate after.

## Eval results

### "Gulian Cup" Ancient Books Named Entity Recognition Evaluation

Second place in the competition. Detailed test results:

| NE Type    | Precision   | Recall | F1    |
|:----------:|:-----------:|:------:|:-----:|
| Book Name  | 77.50       | 73.73  | 75.57 |
| Other Name | 85.85       | 89.32  | 87.55 |
| Micro Avg. | 83.88       | 85.39  | 84.63 |




## About Us

We are from [Datahammer](https://datahammer.net), Beijing Institute of Technology.
For more cooperation, please contact email: ethanyt [at] qq.com

> Created with ❤️ by Tan Yan [![Github icon](https://cdn0.iconfinder.com/data/icons/octicons/1024/mark-github-32.png)](https://github.com/Ethan-yt) and Zewen Chi [![Github icon](https://cdn0.iconfinder.com/data/icons/octicons/1024/mark-github-32.png)](https://github.com/CZWin32768)