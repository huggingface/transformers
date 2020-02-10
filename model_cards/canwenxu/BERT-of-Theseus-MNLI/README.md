---
thumbnail: https://raw.githubusercontent.com/JetRunner/BERT-of-Theseus/master/bert-of-theseus.png
---

# BERT-of-Theseus
See our paper ["BERT-of-Theseus: Compressing BERT by Progressive Module Replacing"](http://arxiv.org/abs/2002.02925).

BERT-of-Theseus is a new compressed BERT by progressively replacing the components of the original BERT.

![BERT of Theseus](https://github.com/JetRunner/BERT-of-Theseus/blob/master/bert-of-theseus.png?raw=true)

## Load Pretrained Model on MNLI

We provide a 6-layer pretrained model on MNLI as a general-purpose model, which can transfer to other sentence classification tasks, outperforming DistillBERT (with the same 6-layer structure) on six tasks of GLUE (dev set).

| Method          | MNLI | MRPC | QNLI | QQP  | RTE  | SST-2 | STS-B |
|-----------------|------|------|------|------|------|-------|-------|
| BERT-base       | 83.5 | 89.5 | 91.2 | 89.8 | 71.1 | 91.5  | 88.9  |
| DistillBERT     | 79.0 | 87.5 | 85.3 | 84.9 | 59.9 | 90.7  | 81.2  |
| BERT-of-Theseus | 82.1 | 87.5 | 88.8 | 88.8 | 70.1 | 91.8  | 87.8  |
