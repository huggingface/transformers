---
language: multilingual

datasets: wikipedia

license: apache-2.0
---

# bert-base-15lang-cased

We are sharing smaller versions of [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) that handle a custom number of languages.

Unlike [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased), our versions give exactly the same representations produced by the original model which preserves the original accuracy.

The measurements below have been computed on a [Google Cloud n1-standard-1 machine (1 vCPU, 3.75 GB)](https://cloud.google.com/compute/docs/machine-types\#n1_machine_type):
|             Model               | Num parameters |   Size   |  Memory  | Loading time |
| ------------------------------- | -------------- | -------- | -------- | ------------ |
| bert-base-multilingual-cased    |   178 million  |  714 MB  | 1400 MB  |    4.2 sec   |
| Geotrend/bert-base-15lang-cased |   141 million  |  564 MB  | 1098 MB  |    3.1 sec   |

Handled languages: en, fr, es, de, zh, ar, ru, vi, el, bg, th, tr, hi, ur and sw.

For more information please visit our paper: [Load What You Need: Smaller Versions of Multilingual BERT](https://www.aclweb.org/anthology/2020.sustainlp-1.16.pdf).

## How to use

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Geotrend/bert-base-15lang-cased")
model = AutoModel.from_pretrained("Geotrend/bert-base-15lang-cased")

```

### How to cite

```bibtex
@inproceedings{smallermbert,
  title={Load What You Need: Smaller Versions of Mutlilingual BERT},
  author={Abdaoui, Amine and Pradel, Camille and Sigel, Grégoire},
  booktitle={SustaiNLP / EMNLP},
  year={2020}
}
```

## Contact 

Please contact amine@geotrend.fr for any question, feedback or request.
