---
language: eo
thumbnail: https://huggingface.co/blog/assets/01_how-to-train/EsperBERTo-thumbnail-v2.png
widget:
- text: "Jen la komenco de bela <mask>."
- text: "Uno du <mask>"
- text: "Jen finiĝas bela <mask>."
---

# EsperBERTo: RoBERTa-like Language model trained on Esperanto

**Companion model to blog post https://huggingface.co/blog/how-to-train** 🔥

## Training Details

- current checkpoint: 566000
- machine name: `galinette`


![](https://huggingface.co/blog/assets/01_how-to-train/EsperBERTo-thumbnail-v2.png)

## Example pipeline

```python
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="julien-c/EsperBERTo-small",
    tokenizer="julien-c/EsperBERTo-small"
)

fill_mask("Jen la komenco de bela <mask>.")

# This is the beginning of a beautiful <mask>.
# =>

# {
#     'score':0.06502299010753632
#     'sequence':'<s> Jen la komenco de bela vivo.</s>'
#     'token':1099
# }
# {
#     'score':0.0421181358397007
#     'sequence':'<s> Jen la komenco de bela vespero.</s>'
#     'token':5100
# }
# {
#     'score':0.024884626269340515
#     'sequence':'<s> Jen la komenco de bela laboro.</s>'
#     'token':1570
# }
# {
#     'score':0.02324388362467289
#     'sequence':'<s> Jen la komenco de bela tago.</s>'
#     'token':1688
# }
# {
#     'score':0.020378097891807556
#     'sequence':'<s> Jen la komenco de bela festo.</s>'
#     'token':4580
# }
```
