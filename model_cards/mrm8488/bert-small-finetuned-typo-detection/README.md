---
language: english
thumbnail:
---

# BERT SMALL + Typo Detection ✍❌✍✔

[BERT SMALL](https://huggingface.co/google/bert_uncased_L-4_H-512_A-8) fine-tuned on [GitHub Typo Corpus](https://github.com/mhagiwara/github-typo-corpus) for **typo detection** (using *NER* style)

## Details of the downstream task (Typo detection as NER)

- Dataset: [GitHub Typo Corpus](https://github.com/mhagiwara/github-typo-corpus) 📚

- [Fine-tune script on NER dataset provided by Huggingface](https://github.com/huggingface/transformers/blob/master/examples/run_ner.py) 🏋️‍♂️

## Metrics on test set 📋

|  Metric   |  # score  |
| :-------: | :-------: |
|    F1     | **89.12** |
| Precision | **93.82** |
|  Recall   | **84.87** |

## Model in action 🔨

Fast usage with **pipelines** 🧪

```python
from transformers import pipeline

typo_checker = pipeline(
    "ner",
    model="mrm8488/bert-small-finetuned-typo-detection",
    tokenizer="mrm8488/bert-small-finetuned-typo-detection"
)

result = typo_checker("here there is an error in coment")
result[1:-1]

# Output:
[{'entity': 'ok', 'score': 0.9021041989326477, 'word': 'here'},
 {'entity': 'ok', 'score': 0.7975626587867737, 'word': 'there'},
 {'entity': 'ok', 'score': 0.8596242070198059, 'word': 'is'},
 {'entity': 'ok', 'score': 0.7071516513824463, 'word': 'an'},
 {'entity': 'ok', 'score': 0.943381130695343, 'word': 'error'},
 {'entity': 'ok', 'score': 0.8047608733177185, 'word': 'in'},
 {'entity': 'ok', 'score': 0.8240702152252197, 'word': 'come'},
 {'entity': 'typo', 'score': 0.5004884004592896, 'word': '##nt'}]
```

It works🎉! we typed ```coment``` instead of ```comment```

Let's try with another example

```python
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
Yeah! We typed wrong ```Add and middleware```


> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
