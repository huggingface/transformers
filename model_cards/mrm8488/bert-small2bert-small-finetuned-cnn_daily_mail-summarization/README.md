---
language: en
license: apache-2.0
datasets:
- cnn_dailymail
tags:
- summarization
---

# Bert-small2Bert-small Summarization with 🤗EncoderDecoder Framework

This model is a warm-started *BERT2BERT* ([small](https://huggingface.co/google/bert_uncased_L-4_H-512_A-8)) model fine-tuned on the *CNN/Dailymail* summarization dataset.

The model achieves a **17.37** ROUGE-2 score on *CNN/Dailymail*'s test dataset.

For more details on how the model was fine-tuned, please refer to 
[this](https://colab.research.google.com/drive/1Ekd5pUeCX7VOrMx94_czTkwNtLN32Uyu?usp=sharing) notebook.

## Results on test set 📝

| Metric | # Value   |
| ------ | --------- |
| **ROUGE-2** | **17.37** |



## Model in Action 🚀

```python
from transformers import BertTokenizerFast, EncoderDecoderModel
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization')
model = EncoderDecoderModel.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization').to(device)

def generate_summary(text):
    # cut off at BERT max length 512
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    output = model.generate(input_ids, attention_mask=attention_mask)

    return tokenizer.decode(output[0], skip_special_tokens=True)
  
text = "your text to be summarized here..."
generate_summary(text)
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
