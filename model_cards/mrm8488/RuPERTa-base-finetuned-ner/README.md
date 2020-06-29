---
language: spanish
thumbnail:
---

# RuPERTa-base  (Spanish RoBERTa) + NER 🎃🏷

This model is a fine-tuned on [NER-C](https://www.kaggle.com/nltkdata/conll-corpora) version of [RuPERTa-base](https://huggingface.co/mrm8488/RuPERTa-base) for **NER** downstream task.

## Details of the downstream task (NER) - Dataset

- [Dataset:  CONLL Corpora ES](https://www.kaggle.com/nltkdata/conll-corpora) 📚

| Dataset                | # Examples |
| ---------------------- | ----- |
| Train                  |  329 K |
| Dev                    | 40 K |


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

## Metrics on evaluation set 🧾

|                                                      Metric                                                       |  # score  |
| :------------------------------------------------------------------------------------: | :-------: |
| F1                                       | **77.55**  
| Precision                                | **75.53** | 
| Recall                                   | **79.68** |    

## Model in action 🔨


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

text ="Julien, CEO de HF, nació en Francia."
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
Julien,: I-PER
CEO: O
de: O
HF,: B-ORG
nació: I-PER
en: I-PER
Francia.: I-LOC
'''
```
Yeah! Not too bad 🎉

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
