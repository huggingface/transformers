---
language:
- japanese
---

binary classification

# Usage
```
print(pipeline("sentiment-analysis",model="daigo/bert-base-japanese-sentiment",tokenizer="daigo/bert-base-japanese-sentiment")("私は幸福である。"))

[{'label': 'ポジティブ', 'score': 0.98430425}]
```
