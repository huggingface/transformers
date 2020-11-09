---
language: multilingual
license: apache-2.0
datasets:
- wikipedia
---

# DistilBERT base multilingual model (cased)

This model is a distilled version of the [BERT base multilingual model](bert-base-multilingual-cased). The code for the distillation process can be found
[here](https://github.com/huggingface/transformers/tree/master/examples/distillation). This model is cased: it does make a difference between english and English.

The model is trained on the concatenation of Wikipedia in 104 different languages listed [here](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages).
The model has 6 layers, 768 dimension and 12 heads, totalizing 134M parameters (compared to 177M parameters for mBERT-base).
On average DistilmBERT is twice as fast as mBERT-base.

We encourage to check [BERT base multilingual model](bert-base-multilingual-cased) to know more about usage, limitations and potential biases.

| Model                        | English | Spanish | Chinese | German | Arabic  | Urdu |
| :---:                        | :---:   | :---:   | :---:   | :---:  | :---:   | :---:|
| mBERT base cased (computed)  | 82.1    | 74.6    | 69.1    | 72.3   | 66.4    | 58.5 |
| mBERT base uncased (reported)| 81.4    | 74.3    | 63.8    | 70.5   | 62.1    | 58.3 |
| DistilmBERT                  | 78.2    | 69.1    | 64.0    | 66.3   | 59.1    | 54.7 |

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
