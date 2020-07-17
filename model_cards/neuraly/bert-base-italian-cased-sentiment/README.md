---
language: it
thumbnail: https://neuraly.ai/static/assets/images/huggingface/thumbnail.png
tags:
- sentiment
- Italian
license: MIT
widget:
- text: "Huggingface è un team fantastico!"
---

# 🤗 + neuraly - Italian BERT Sentiment model
  
## Model description  
  
This model performs sentiment analysis on Italian sentences. It was trained starting from an instance of [bert-base-italian-cased](https://huggingface.co/dbmdz/bert-base-italian-cased), and fine-tuned on an Italian dataset of tweets, reaching 82% of accuracy on the latter one.
  
## Intended uses & limitations  
  
#### How to use  
  
```python
import torch
from torch import nn  
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("neuraly/bert-base-italian-cased-sentiment")
# Load the model, use .cuda() to load it on the GPU
model = AutoModelForSequenceClassification.from_pretrained("neuraly/bert-base-italian-cased-sentiment")

sentence = 'Huggingface è un team fantastico!'
input_ids = tokenizer.encode(sentence, add_special_tokens=True)

# Create tensor, use .cuda() to transfer the tensor to GPU
tensor = torch.tensor(input_ids).long()
# Fake batch dimension
tensor = tensor.unsqueeze(0)

# Call the model and get the logits
logits, = model(tensor)

# Remove the fake batch dimension
logits = logits.squeeze(0)

# The model was trained with a Log Likelyhood + Softmax combined loss, hence to extract probabilities we need a softmax on top of the logits tensor
proba = nn.functional.softmax(logits, dim=0)

# Unpack the tensor to obtain negative, neutral and positive probabilities
negative, neutral, positive = proba
```  
  
#### Limitations and bias  
  
A possible drawback (or bias) of this model is related to the fact that it was trained on a tweet dataset, with all the limitations that come with it. The domain is strongly related to football players and teams, but it works surprisingly well even on other topics.
  
## Training data  
  
We trained the model by combining the two tweet datasets taken from [Sentipolc EVALITA 2016](http://www.di.unito.it/~tutreeb/sentipolc-evalita16/data.html).  Overall the dataset consists of  45K pre-processed tweets.

The model weights come from a pre-trained instance of [bert-base-italian-cased](https://huggingface.co/dbmdz/bert-base-italian-cased). A huge "thank you" goes to that team, brilliant work!
  
## Training procedure  
  
#### Preprocessing

We tried to save as much information as possible, since BERT captures extremely well the semantic of complex text sequences. Overall we removed only **@mentions**, **urls** and **emails** from every tweet and kept pretty much everything else.

#### Hardware

- **GPU**: Nvidia GTX1080ti
- **CPU**: AMD Ryzen7 3700x 8c/16t
- **RAM**: 64GB DDR4

#### Hyperparameters

- Optimizer: **AdamW** with learning rate of **2e-5**, epsilon of **1e-8**
- Max epochs: **5**
- Batch size: **32**
- Early Stopping: **enabled** with patience = 1

Early stopping was triggered after 3 epochs.
  
## Eval results 

The model achieves an overall accuracy on the test set equal to 82%
The test set is a 20% split of the whole dataset.

## About us
[Neuraly](https://neuraly.ai) is a young and dynamic startup committed to designing AI-driven solutions and services through the most advanced Machine Learning and Data Science technologies. You can find out more about who we are and what we do on our [website](https://neuraly.ai).

## Acknowledgments

Thanks to the generous support from the [Hugging Face](https://huggingface.co/) team,
it is possible to download the model from their S3 storage and live test it from their inference API 🤗.
