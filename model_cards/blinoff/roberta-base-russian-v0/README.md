---
language: ru
widget:
- text: "Мозг — это машина <mask>, которая пытается снизить ошибку в прогнозе."
---

# RoBERTa-like language model trained on part of part of TAIGA corpus

## Training Details

- about 60k steps

![]()

## Example pipeline

```python
from transformers import pipeline
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained('blinoff/roberta-base-russian-v0', max_len=512)

fill_mask = pipeline(
    "fill-mask",
    model="blinoff/roberta-base-russian-v0",
    tokenizer=tokenizer
)

fill_mask("Мозг — это машина <mask>, которая пытается снизить ошибку в прогнозе.")

# {
#     'sequence': '<s>Мозг — это машина города, которая пытается снизить ошибку в прогнозе.</s>',
#     'score': 0.012859329581260681,
#     'token': 2144,
#     'token_str': 'ĠÐ³Ð¾ÑĢÐ¾Ð´Ð°'
# },
# {
#     'sequence': '<s>Мозг — это машина человека, которая пытается снизить ошибку в прогнозе.</s>',
#     'score': 0.01185101643204689,
#     'token': 1470,
#     'token_str': 'ĠÑĩÐµÐ»Ð¾Ð²ÐµÐºÐ°'
# },
# {
#     'sequence': '<s>Мозг — это машина дома, которая пытается снизить ошибку в прогнозе.</s>',
#     'score': 0.009940559044480324,
#     'token': 1411,
#     'token_str': 'ĠÐ´Ð¾Ð¼Ð°'
# },
# {
#     'sequence': '<s>Мозг — это машина женщина, которая пытается снизить ошибку в прогнозе.</s>',
#     'score': 0.007794599514454603,
#     'token': 2707,
#     'token_str': 'ĠÐ¶ÐµÐ½ÑīÐ¸Ð½Ð°'
# },
# {
#     'sequence': '<s>Мозг — это машина женщины, которая пытается снизить ошибку в прогнозе.</s>',
#     'score': 0.007725382689386606,
#     'token': 3546,
#     'token_str': 'ĠÐ¶ÐµÐ½ÑīÐ¸Ð½Ñĭ'
# }
```
