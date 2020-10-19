---
thumbnail: https://huggingface.co/front/thumbnails/microsoft.png
license: mit
---

## DeBERTa: Decoding-enhanced BERT with Disentangled Attention

[DeBERTa](https://arxiv.org/abs/2006.03654) improves the BERT and RoBERTa models using disentangled attention and enhanced mask decoder. With those two improvements, DeBERTa out perform RoBERTa on a majority of NLU tasks with 80GB training data. 

Please check the [official repository](https://github.com/microsoft/DeBERTa) for more details and updates.


#### Fine-tuning on NLU tasks

We present the dev results on SQuAD 1.1/2.0 and several GLUE benchmark tasks.

| Model             | SQuAD 1.1 | SQuAD 2.0 | MNLI-m | SST-2 | QNLI | CoLA | RTE  | MRPC | QQP  |STS-B|
|-------------------|-----------|-----------|--------|-------|------|------|------|------|------|-----|
| BERT-Large        | 90.9/84.1 | 81.8/79.0 | 86.6   | 93.2  | 92.3 | 60.6 | 70.4 | 88.0 | 91.3 |90.0 |
| RoBERTa-Large     | 94.6/88.9 | 89.4/86.5 | 90.2   | 96.4  | 93.9 | 68.0 | 86.6 | 90.9 | 92.2 |92.4 |
| XLNet-Large       | 95.1/89.7 | 90.6/87.9 | 90.8   | 97.0  | 94.9 | 69.0 | 85.9 | 90.8 | 92.3 |92.5 |
| **DeBERTa-Large** | 95.5/90.1 | 90.7/88.0 | 91.1   | 96.5  | 95.3 | 69.5 | 88.1 | 92.5 | 92.3 |92.5 |

### Citation

If you find DeBERTa useful for your work, please cite the following paper:

``` latex
@misc{he2020deberta,
    title={DeBERTa: Decoding-enhanced BERT with Disentangled Attention},
    author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
    year={2020},
    eprint={2006.03654},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
		}
```
