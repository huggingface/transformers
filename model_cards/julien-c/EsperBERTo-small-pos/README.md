---
language: esperanto
thumbnail: https://huggingface.co/blog/assets/EsperBERTo-thumbnail-v2.png
widget:
- text: "Mi estas viro kej estas tago varma."
---

# EsperBERTo: RoBERTa-like Language model trained on Esperanto

**Companion model to blog post https://huggingface.co/blog/how-to-train** 🔥

## Training Details

- current checkpoint: 566000
- machine name: `galinette`


![](https://huggingface.co/blog/assets/EsperBERTo-thumbnail-v2.png)

## Example pipeline

```python
from transformers import TokenClassificationPipeline, pipeline


MODEL_PATH = "./models/EsperBERTo-small-pos/"

nlp = pipeline(
    "ner",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
)
# or instantiate a TokenClassificationPipeline directly.

nlp("Mi estas viro kej estas tago varma.")

# {'entity': 'PRON', 'score': 0.9979867339134216, 'word': ' Mi'}
# {'entity': 'VERB', 'score': 0.9683094620704651, 'word': ' estas'}
# {'entity': 'VERB', 'score': 0.9797462821006775, 'word': ' estas'}
# {'entity': 'NOUN', 'score': 0.8509314060211182, 'word': ' tago'}
# {'entity': 'ADJ', 'score': 0.9996201395988464, 'word': ' varma'}
```