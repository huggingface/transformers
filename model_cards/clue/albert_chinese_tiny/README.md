---
language: chinese
---

## albert_chinese_tiny

### Overview

**Language model:** albert-tiny
**Model size:** 16M
**Language:** Chinese
**Training data:** [CLUECorpusSmall](https://github.com/CLUEbenchmark/CLUECorpus2020)
**Eval data:** [CLUE dataset](https://github.com/CLUEbenchmark/CLUE)

### Results

For results on downstream tasks like text classification, please refer to [this repository](https://github.com/CLUEbenchmark/CLUE).

### Usage

**NOTE:**Since sentencepiece is not used in `albert_chinese_tiny` model, you have to call **BertTokenizer** instead of AlbertTokenizer !!!

```
import torch
from transformers import BertTokenizer, AlbertModel
tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")
albert = AlbertModel.from_pretrained("clue/albert_chinese_tiny")
```

### About CLUE benchmark

Organization of Language Understanding Evaluation benchmark for Chinese: tasks & datasets, baselines, pre-trained Chinese models, corpus and leaderboard.

Github: https://github.com/CLUEbenchmark
Website: https://www.cluebenchmarks.com/
