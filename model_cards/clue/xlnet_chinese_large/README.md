---
language: chinese
---

## xlnet_chinese_large

### Overview

**Language model:** xlnet-large
**Model size:** 1.3G
**Language:** Chinese
**Training data:** [CLUECorpusSmall](https://github.com/CLUEbenchmark/CLUECorpus2020)
**Eval data:** [CLUE dataset](https://github.com/CLUEbenchmark/CLUE)

### Results

For results on downstream tasks like text classification, please refer to [this repository](https://github.com/CLUEbenchmark/CLUE).

### Usage

```
import torch
from transformers import XLNetTokenizer,XLNetModel
tokenizer = XLNetTokenizer.from_pretrained("clue/xlnet_chinese_large")
xlnet = XLNetModel.from_pretrained("clue/xlnet_chinese_large")
```

### About CLUE benchmark

Organization of Language Understanding Evaluation benchmark for Chinese: tasks & datasets, baselines, pre-trained Chinese models, corpus and leaderboard.

Github: https://github.com/CLUEbenchmark
Website: https://www.cluebenchmarks.com/
