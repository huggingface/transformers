---
language: french
---

# camembert-large-fquad

## Description

A native French Question Answering model [CamemBERT-large](https://camembert-model.fr/) fine-tuned on [FQuAD](https://fquad.illuin.tech/).

## FQuAD Leaderboard and evaluation scores

The results of Camembert-large-fquad can be compared with other state-of-the-art models of the [FQuAD Leaderboard](https://illuin-tech.github.io/FQuAD-explorer/).

On the test set the model scores,

```shell
{"f1": 91.5, "exact_match": 82.0}
```

On the development set the model scores,

```shell
{"f1": 91.0, "exact_match": 81.2}
```

Note : You can also explore the results of the model on [FQuAD-Explorer](https://illuin-tech.github.io/FQuAD-explorer/) !

## Usage

```python
from transformers import pipeline

nlp = pipeline('question-answering', model='illuin/camembert-large-fquad', tokenizer='illuin/camembert-large-fquad')

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
