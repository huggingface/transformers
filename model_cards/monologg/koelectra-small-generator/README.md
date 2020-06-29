---
language: Korean
---

# KoELECTRA (Small Generator)

Pretrained ELECTRA Language Model for Korean (`koelectra-small-generator`)

For more detail, please see [original repository](https://github.com/monologg/KoELECTRA/blob/master/README_EN.md).

## Usage

### Load model and tokenizer

```python
>>> from transformers import ElectraModel, ElectraTokenizer

>>> model = ElectraModel.from_pretrained("monologg/koelectra-small-generator")
>>> tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-generator")
```

### Tokenizer example

```python
>>> from transformers import ElectraTokenizer
>>> tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-generator")
>>> tokenizer.tokenize("[CLS] 한국어 ELECTRA를 공유합니다. [SEP]")
['[CLS]', '한국어', 'E', '##L', '##EC', '##T', '##RA', '##를', '공유', '##합니다', '.', '[SEP]']
>>> tokenizer.convert_tokens_to_ids(['[CLS]', '한국어', 'E', '##L', '##EC', '##T', '##RA', '##를', '공유', '##합니다', '.', '[SEP]'])
[2, 18429, 41, 6240, 15229, 6204, 20894, 5689, 12622, 10690, 18, 3]
```

## Example using ElectraForMaskedLM

```python
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="monologg/koelectra-small-generator",
    tokenizer="monologg/koelectra-small-generator"
)

print(fill_mask("나는 {} 밥을 먹었다.".format(fill_mask.tokenizer.mask_token)))
```
