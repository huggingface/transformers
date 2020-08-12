---
language: zu
---

# zuBERTa
zuBERTa is a RoBERTa style transformer language model trained on zulu text.

## Intended uses & limitations
The model can be used for getting embeddings to use on a down-stream task such as question answering.

#### How to use

```python
>>> from transformers import pipeline
>>> from transformers import AutoTokenizer, AutoModelWithLMHead

>>> tokenizer = AutoTokenizer.from_pretrained("MoseliMotsoehli/zuBERTa")
>>> model = AutoModelWithLMHead.from_pretrained("MoseliMotsoehli/zuBERTa")
>>> unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
>>> unmasker("Abafika eNkandla bafika sebeholwa <mask> uMpongo kaZingelwayo.")

[
  {
    "sequence": "<s>Abafika eNkandla bafika sebeholwa khona uMpongo kaZingelwayo.</s>",
    "score": 0.050459690392017365,
    "token": 555,
    "token_str": "Ġkhona"
  },
  {
    "sequence": "<s>Abafika eNkandla bafika sebeholwa inkosi uMpongo kaZingelwayo.</s>",
    "score": 0.03668094798922539,
    "token": 2321,
    "token_str": "Ġinkosi"
  },
  {
    "sequence": "<s>Abafika eNkandla bafika sebeholwa ubukhosi uMpongo kaZingelwayo.</s>",
    "score": 0.028774697333574295,
    "token": 5101,
    "token_str": "Ġubukhosi"
  }
]
```

## Training data

1. 30k sentences of text, came from the [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download) of zulu 2018. These were collected from news articles and creative writtings. 
2. ~7500 articles of human generated translations were scraped from the zulu [wikipedia](https://zu.wikipedia.org/wiki/Special:AllPages).

### BibTeX entry and citation info

```bibtex
@inproceedings{author = {Moseli Motsoehli},
  title = {Towards transformation of Southern African language models through transformers.},
  year={2020}
}
```
