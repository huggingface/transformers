# DistilRoBERTa + Sentiment Analysis 😂😢😡😃😯

This in an adapted version of [@omarsar0](https://twitter.com/omarsar0) [tutorial](https://t.co/WMnATW0Hwf?amp=1)
He explains everything so detailed and provided the dataset. I just changed some parameters and created the ```config.json```file to upload it to [🤗Transformers HUB](https://huggingface.co/) 


In this tutorial, he shows how to fine-tune a language model (LM) for **emotion classification** with code adapted from this [tutorial](https://zablo.net/blog/post/custom-classifier-on-bert-model-guide-polemo2-sentiment-analysis/) by MARCIN ZABŁOCKI. 

The emotions covered are:
 - sadness 😢
 - joy 😃
 - love 🥰
 - anger 😡
 - fear 😱
 - surprise 😯

## Details of the language model
The base model used is [DistilRoBERTa](https://huggingface.co/distilroberta-base)

## Details of the downstream task (Sentence classification) - Dataset 📚

| Dataset split               | # Size | # Sequences |
| ---------------------- | ----- | ------|
|Train                   | 1.58M | 20000
| Validation                | 200 KB |
| Test                      | 202 KB |


## Results after training 🏋️‍♀️🧾

|emotion |precision    |recall|  f1-score|   support|
|-------|-------------|------|----------|----------|
|sadness| 0.973868  |0.949066  |0.961307|      589|
|joy   |0.970313  |0.901306  |0.934537|       689|
|love   |0.743119  |0.925714  |0.824427|       175|    
|anger  | 0.884615|  0.969349|  0.925046|       261|      
|fear   |0.951456  |0.875000|  0.911628|       224|      
|surprise|   0.750000|  0.919355|  0.826087|    62|
|         | | | | |
|**accuracy**| | |                  0.924000|      2000|
|**macro avg**|   0.878895|  0.923298|  0.897172|      2000|
|**weighted avg**|   0.931355|  0.924000|  0.925620|      2000|

## Model in action 🔨

Fast usage with **pipelines** 🧪

```python
from transformers import pipeline

nlp_sentiment = pipeline(
    "sentiment-analysis",
    model="mrm8488/distilroberta-base-finetuned-sentiment",
    tokenizer="mrm8488/distilroberta-base-finetuned-sentiment"
)

text = "i feel i should return to the start of the weekend so my loyal readers can get a feeling for things up to this point"

nlp_sentiment(text)
# Output: [{'label': 'love', 'score': 0.2183746}]
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
