---
language: en
tags:
- exbert

license: apache-2.0
datasets:
- openwebtext
---

# DistilRoBERTa base model

This model is a distilled version of the [RoBERTa-base model](https://huggingface.co/roberta-base). It follows the same training procedure as [DistilBERT](https://huggingface.co/distilbert-base-uncased).
The code for the distillation process can be found [here](https://github.com/huggingface/transformers/tree/master/examples/distillation).
This model is case-sensitive: it makes a difference between english and English.

The model has 6 layers, 768 dimension and 12 heads, totalizing 82M parameters (compared to 125M parameters for RoBERTa-base).
On average DistilRoBERTa is twice as fast as Roberta-base.

We encourage to check [RoBERTa-base model](https://huggingface.co/roberta-base) to know more about usage, limitations and potential biases.

## Training data

DistilRoBERTa was pre-trained on [OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/), a reproduction of OpenAI's WebText dataset (it is ~4 times less training data than the teacher RoBERTa).

## Evaluation results

When fine-tuned on downstream tasks, this model achieves the following results:

Glue test results:

| Task | MNLI | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  |
|:----:|:----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|
|      | 84.0 | 89.4 | 90.8 | 92.5  | 59.3 | 88.3  | 86.6 | 67.9 |

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

<a href="https://huggingface.co/exbert/?model=distilroberta-base">
	<img width="300px" src="https://cdn-media.huggingface.co/exbert/button.png">
</a>
