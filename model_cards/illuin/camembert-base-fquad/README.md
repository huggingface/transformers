---
language: french
---

# camembert-base-fquad

## Description

A native French Question Answering model [CamemBERT-base](https://camembert-model.fr/) fine-tuned on [FQuAD](https://fquad.illuin.tech/).

## Evaluation results

On the development set.

```shell
{"f1": 88.1, "exact_match": 78.1}
```

On the test set.

```shell
{"f1": 88.3, "exact_match": 78.0}
```

## Usage

```python
from transformers import pipeline

nlp = pipeline('question-answering', model='illuin/camembert-base-fquad', tokenizer='illuin/camembert-base-fquad')

nlp({
    'question': "Qui est Claude Monet?",
    'context': "Claude Monet, né le 14 novembre 1840 à Paris et mort le 5 décembre 1926 à Giverny, est un peintre français et l’un des fondateurs de l'impressionnisme."
})
```

## Citation

If you use our work, please cite:

```bibtex
@article{dHoffschmidt2020FQuADFQ,
  title={FQuAD: French Question Answering Dataset},
  author={Martin d'Hoffschmidt and Maxime Vidal and Wacim Belblidia and Tom Brendl'e and Quentin Heinrich},
  journal={ArXiv},
  year={2020},
  volume={abs/2002.06071}
}
```
