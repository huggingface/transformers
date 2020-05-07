---
language: spanish
thumbnail:
---

# Spanish TinyBERT + NER

This model is a fine-tuned on [NER-C](https://www.kaggle.com/nltkdata/conll-corpora) of a [Spanish Tiny Bert](https://huggingface.co/mrm8488/es-tinybert-v1-1) model I created using *distillation* for **NER** downstream task. The **size** of the model is **55MB**

## Details of the downstream task (NER) - Dataset

- [Dataset:  CONLL Corpora ES](https://www.kaggle.com/nltkdata/conll-corpora) 

I preprocessed the dataset and splitted it as train / dev (80/20)

| Dataset                | # Examples |
| ---------------------- | ----- |
| Train                  | 8.7 K |
| Dev                    | 2.2 K |


- [Fine-tune on NER script provided by Huggingface](https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py)

- Labels covered:

```
B-LOC
B-MISC
B-ORG
B-PER
I-LOC
I-MISC
I-ORG
I-PER
O
```

## Metrics on evaluation set:

|                                                      Metric                                                       |  # score  |
| :------------------------------------------------------------------------------------: | :-------: |
| F1                                       | **70.00**  
| Precision                                | **67.83** | 
| Recall                                   | **71.46** |    

## Comparison:

|                                                      Model                                                       |  # F1 score  |Size(MB)|
| :--------------------------------------------------------------------------------------------------------------: | :-------: |:------|
|                                        bert-base-spanish-wwm-cased (BETO)                                        |   88.43   | 421
| [bert-spanish-cased-finetuned-ner](https://huggingface.co/mrm8488/bert-spanish-cased-finetuned-ner) | **90.17** | 420 |
|                                              Best Multilingual BERT                                              |   87.38   | 681 |
|TinyBERT-spanish-uncased-finetuned-ner (this one)                                                                  | 70.00 | **55** |

## Model in action


Example of usage:

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

id2label = {
    "0": "B-LOC",
    "1": "B-MISC",
    "2": "B-ORG",
    "3": "B-PER",
    "4": "I-LOC",
    "5": "I-MISC",
    "6": "I-ORG",
    "7": "I-PER",
    "8": "O"
}

tokenizer = AutoTokenizer.from_pretrained('mrm8488/TinyBERT-spanish-uncased-finetuned-ner')
model = AutoModelForTokenClassification.from_pretrained('mrm8488/TinyBERT-spanish-uncased-finetuned-ner')
text ="Mis amigos están pensando viajar a Londres este verano."
input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

outputs = model(input_ids)
last_hidden_states = outputs[0]

for m in last_hidden_states:
  for index, n in enumerate(m):
    if(index > 0 and index <= len(text.split(" "))):
      print(text.split(" ")[index-1] + ": " + id2label[str(torch.argmax(n).item())])
      
'''
Output:
--------
Mis: O
amigos: O
están: O
pensando: O
viajar: O
a: O
Londres: B-LOC
este: O
verano.: O
'''
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
