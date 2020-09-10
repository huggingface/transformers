---
thumbnail: https://huggingface.co/front/thumbnails/microsoft.png
tags:
- text-classification
license: mit
---

## MiniLM: Small and Fast Pre-trained Models for Language Understanding and Generation

MiniLM is a distilled model from the paper "[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)".

Please find the information about preprocessing, training and full details of the MiniLM in the [original MiniLM repository](https://github.com/microsoft/unilm/blob/master/minilm/).

Please note: This checkpoint can be an inplace substitution for BERT and it needs to be fine-tuned before use!

### English Pre-trained Models
We release the **uncased** **12**-layer model with **384** hidden size distilled from an in-house pre-trained [UniLM v2](/unilm) model in BERT-Base size.

- MiniLMv1-L12-H384-uncased: 12-layer, 384-hidden, 12-heads, 33M parameters, 2.7x faster than BERT-Base

#### Fine-tuning on NLU tasks

We present the dev results on SQuAD 2.0 and several GLUE benchmark tasks.

| Model                                             | #Param | SQuAD 2.0 | MNLI-m | SST-2 | QNLI | CoLA | RTE  | MRPC | QQP  |
|---------------------------------------------------|--------|-----------|--------|-------|------|------|------|------|------|
| [BERT-Base](https://arxiv.org/pdf/1810.04805.pdf) | 109M   | 76.8      | 84.5   | 93.2  | 91.7 | 58.9 | 68.6 | 87.3 | 91.3 |
| **MiniLM-L12xH384**                               | 33M    | 81.7      | 85.7   | 93.0  | 91.5 | 58.5 | 73.3 | 89.5 | 91.3 |

### Citation

If you find MiniLM useful in your research, please cite the following paper:

``` latex
@misc{wang2020minilm,
    title={MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers},
    author={Wenhui Wang and Furu Wei and Li Dong and Hangbo Bao and Nan Yang and Ming Zhou},
    year={2020},
    eprint={2002.10957},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
