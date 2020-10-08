---
language: es
thumbnail: https://i.imgur.com/DUlT077.jpg
widget:
- text: "España es un país muy <mask> en la UE"
---

# RuPERTa: the Spanish RoBERTa 🎃<img src="https://abs-0.twimg.com/emoji/v2/svg/1f1ea-1f1f8.svg" alt="spain flag" width="25"/>

RuPERTa-base (uncased) is a [RoBERTa model](https://github.com/pytorch/fairseq/tree/master/examples/roberta) trained on a *uncased* verison of [big Spanish corpus](https://github.com/josecannete/spanish-corpora).
RoBERTa iterates on BERT's pretraining procedure, including training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data.
The architecture is the same as `roberta-base`:

`roberta.base:` **RoBERTa** using the **BERT-base architecture 125M** params

## Benchmarks 🧾 
WIP (I continue working on it) 🚧

| Task/Dataset     |    F1 | Precision | Recall |                                                                        Fine-tuned model |                                                                                                                                                                                                                                                                                               Reproduce it |
| -------- | ----: | --------: | -----: | --------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| POS      | 97.39 |     97.47 |  97.32 | [RuPERTa-base-finetuned-pos](https://huggingface.co/mrm8488/RuPERTa-base-finetuned-pos) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrm8488/shared_colab_notebooks/blob/master/RuPERTa_base_finetuned_POS.ipynb)
| NER      | 77.55 |     75.53 |  79.68 | [RuPERTa-base-finetuned-ner](https://huggingface.co/mrm8488/RuPERTa-base-finetuned-ner) |
| SQUAD-es v1 |  to-do |       |    |[RuPERTa-base-finetuned-squadv1](https://huggingface.co/mrm8488/RuPERTa-base-finetuned-squadv1)
| SQUAD-es v2 |  to-do |       |  |[RuPERTa-base-finetuned-squadv2](https://huggingface.co/mrm8488/RuPERTa-base-finetuned-squadv2)

## Model in action 🔨

### Usage for POS and NER 🏷

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

tokenizer = AutoTokenizer.from_pretrained('mrm8488/RuPERTa-base-finetuned-ner')
model = AutoModelForTokenClassification.from_pretrained('mrm8488/RuPERTa-base-finetuned-ner')

text ="Julien, CEO de HF, nació en Francia."

input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

outputs = model(input_ids)
last_hidden_states = outputs[0]

for m in last_hidden_states:
  for index, n in enumerate(m):
    if(index > 0 and index <= len(text.split(" "))):
      print(text.split(" ")[index-1] + ": " + id2label[str(torch.argmax(n).item())])

# Output:
'''
Julien,: I-PER
CEO: O
de: O
HF,: B-ORG
nació: I-PER
en: I-PER
Francia.: I-LOC
'''
```

For **POS** just change the `id2label` dictionary and the model path to [mrm8488/RuPERTa-base-finetuned-pos](https://huggingface.co/mrm8488/RuPERTa-base-finetuned-pos)

### Fast usage for LM with `pipelines` 🧪

```python
from transformers import AutoModelWithLMHead, AutoTokenizer
model = AutoModelWithLMHead.from_pretrained('mrm8488/RuPERTa-base')
tokenizer = AutoTokenizer.from_pretrained("mrm8488/RuPERTa-base", do_lower_case=True)

from transformers import pipeline

pipeline_fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

pipeline_fill_mask("España es un país muy <mask> en la UE")
```

```json
[
  {
    "score": 0.1814306527376175,
    "sequence": "<s> españa es un país muy importante en la ue</s>",
    "token": 1560
  },
  {
    "score": 0.024842597544193268,
    "sequence": "<s> españa es un país muy fuerte en la ue</s>",
    "token": 2854
  },
  {
    "score": 0.02473250962793827,
    "sequence": "<s> españa es un país muy pequeño en la ue</s>",
    "token": 2948
  },
  {
    "score": 0.023991240188479424,
    "sequence": "<s> españa es un país muy antiguo en la ue</s>",
    "token": 5240
  },
  {
    "score": 0.0215945765376091,
    "sequence": "<s> españa es un país muy popular en la ue</s>",
    "token": 5782
  }
]
```

## Acknowledgments

I thank [🤗/transformers team](https://github.com/huggingface/transformers) for answering my doubts and Google for helping me with the [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc) program.

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
