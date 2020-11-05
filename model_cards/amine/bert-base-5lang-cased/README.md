---
language: 
- en
- fr
- es
- de
- zh

tags:
- pytorch
- bert
- multilingual
- en
- fr
- es
- de
- zh

datasets: wikipedia

license: apache-2.0

inference: false
---

# bert-base-5lang-cased
This is a smaller version of [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) that handles only 5 languages (en, fr, es, de and zh) instead of 104.
The model is therefore 30% smaller than the original one (124M parameters instead of 178M) but gives exactly the same representations for the above cited languages. 
Starting from `bert-base-5lang-cased` will facilitate the deployment of your model on public cloud platforms while keeping similar results. 
For instance, Google Cloud Platform requires that the model size on disk should be lower than 500 MB for serveless deployments (Cloud Functions / Cloud ML) which is not the case of the original `bert-base-multilingual-cased`.

For more information about the models size, memory footprint and loading time please refer to the table below:

|            Model             | Num parameters |   Size   |  Memory  | Loading time |
| ---------------------------- | -------------- | -------- | -------- | ------------ |
| bert-base-multilingual-cased |   178 million  |  714 MB  | 1400 MB  |    4.2 sec   |
| bert-base-5lang-cased        |   124 million  |  495 MB  |  950 MB  |    3.6 sec   |

These measurements have been computed on a [Google Cloud n1-standard-1 machine (1 vCPU, 3.75 GB)](https://cloud.google.com/compute/docs/machine-types\#n1_machine_type).

## How to use

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("amine/bert-base-5lang-cased")
model = AutoModel.from_pretrained("amine/bert-base-5lang-cased")

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