---
language: chinese
---

## roberta_chinese_base

### Overview

**Language model:** roberta-base
**Model size:** 392M
**Language:** Chinese
**Training data:** [CLUECorpusSmall](https://github.com/CLUEbenchmark/CLUECorpus2020)
**Eval data:** [CLUE dataset](https://github.com/CLUEbenchmark/CLUE)

### Results

For results on downstream tasks like text classification, please refer to [this repository](https://github.com/CLUEbenchmark/CLUE).

### Usage

**NOTE:** You have to call **BertTokenizer** instead of RobertaTokenizer !!!

```
import torch
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_base")
roberta = BertModel.from_pretrained("clue/roberta_chinese_base")
```

### About CLUE benchmark

Organization of Language Understanding Evaluation benchmark for Chinese: tasks & datasets, baselines, pre-trained Chinese models, corpus and leaderboard.

Github: https://github.com/CLUEbenchmark
Website: https://www.cluebenchmarks.com/
