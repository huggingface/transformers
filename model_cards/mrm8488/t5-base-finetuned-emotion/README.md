---
language: english
---

# T5-base fine-tuned for Emotion Recognition 😂😢😡😃😯


[Google's T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) base fine-tuned on [emotion recognition](https://github.com/dair-ai/emotion_dataset) dataset for **Emotion Recognition** downstream task.

## Details of T5

The **T5** model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu* in Here the abstract:

Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code.

![model image](https://i.imgur.com/jVFMMWR.png)

## Details of the downstream task (Sentiment Recognition) - Dataset 📚

[Elvis Saravia](https://twitter.com/omarsar0) has gathered a great [dataset](https://github.com/dair-ai/emotion_dataset) for emotion recognition. It allows to classifiy the text into one of the following **6** emotions:

 - sadness 😢
 - joy 😃
 - love 🥰
 - anger 😡
 - fear 😱
 - surprise 😯

## Model fine-tuning 🏋️‍

The training script is a slightly modified version of [this Colab Notebook](https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb) created by [Suraj Patil](https://github.com/patil-suraj), so all credits to him!

## Test set metrics 🧾

|          |precision | recall  | f1-score |support|
|----------|----------|---------|----------|-------|
|anger     |      0.93|     0.92|      0.93|    275|
|fear      |      0.91|     0.87|      0.89|    224|
|joy       |      0.97|     0.94|      0.95|    695|
|love      |      0.80|     0.91|      0.85|    159|
|sadness   |      0.97|     0.97|      0.97|    521|
|surpirse  |      0.73|     0.89|      0.80|     66|
|                                                  |
|accuracy|            |         |      0.93|   2000|
|macro avg|       0.89|     0.92|      0.90|   2000|
|weighted avg|    0.94|     0.93|      0.93|   2000|
    
    
                 
    



## Model in Action 🚀

```python
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")

model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

def get_emotion(text):
  input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

  output = model.generate(input_ids=input_ids,
               max_length=2)
  
  dec = [tokenizer.decode(ids) for ids in output]
  label = dec[0]
  return label
  
 get_emotion("i feel as if i havent blogged in ages are at least truly blogged i am doing an update cute") # Output: 'joy'
 
 get_emotion("i have a feeling i kinda lost my best friend") # Output: 'sadness'
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
