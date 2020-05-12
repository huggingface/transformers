---
language: multilingual
thumbnail:
---

# DISTILBERT 🌎 + Typo Detection ✍❌✍✔

[distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased) fine-tuned on [GitHub Typo Corpus](https://github.com/mhagiwara/github-typo-corpus) for **typo detection** (using *NER* style)

## Details of the downstream task (Typo detection as NER)

- Dataset: [GitHub Typo Corpus](https://github.com/mhagiwara/github-typo-corpus) 📚 for 15 languages

- [Fine-tune script on NER dataset provided by Huggingface](https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py) 🏋️‍♂️

## Metrics on test set 📋

|  Metric   |  # score  |
| :-------: | :-------: |
|    F1     | **93.51** |
| Precision | **96.08** |
|  Recall   | **91.06** |

## Model in action 🔨

Fast usage with **pipelines** 🧪

```python
from transformers import pipeline

typo_checker = pipeline(
    "ner",
    model="mrm8488/distilbert-base-multi-cased-finetuned-typo-detection",
    tokenizer="mrm8488/distilbert-base-multi-cased-finetuned-typo-detection"
)

result = typo_checker("Adddd validation midelware")
result[1:-1]

# Output:
[{'entity': 'ok', 'score': 0.7128152847290039, 'word': 'add'},
 {'entity': 'typo', 'score': 0.5388424396514893, 'word': '##dd'},
 {'entity': 'ok', 'score': 0.94792640209198, 'word': 'validation'},
 {'entity': 'typo', 'score': 0.5839331746101379, 'word': 'mid'},
 {'entity': 'ok', 'score': 0.5195121765136719, 'word': '##el'},
 {'entity': 'ok', 'score': 0.7222476601600647, 'word': '##ware'}]
```
It works🎉! We typed wrong ```Add and middleware```


> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
