# How to use

```
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-BERT-char16424", do_lower_case=False)
model = AutoModel.from_pretrained("snunlp/KR-BERT-char16424")
```
