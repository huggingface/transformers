---
language: en
license: apache-2.0
datasets:
- bookcorpus
- wikipedia
---

# DistilBERT base model (cased)

This model is a distilled version of the [BERT base model](https://huggingface.co/bert-base-cased).
It was introduced in [this paper](https://arxiv.org/abs/1910.01108).
The code for the distillation process can be found
[here](https://github.com/huggingface/transformers/tree/master/examples/distillation).
This model is cased: it does make a difference between english and English.

All the training details on the pre-training, the uses, limitations and potential biases are the same as for [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased).
We highly encourage to check it if you want to know more.

## Evaluation results

When fine-tuned on downstream tasks, this model achieves the following results:

Glue test results:

| Task | MNLI | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  |
|:----:|:----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|
|      | 81.5 | 87.8 | 88.2 | 90.4  | 47.2 | 85.5  | 85.6 | 60.6 |

### BibTeX entry and citation info

```bibtex
@article{Sanh2019DistilBERTAD,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.01108}
}
```
