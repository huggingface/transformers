---
language: italian
thumbnail:
---

# Italian BERT fine-tuned on SQuAD_it v1

[Italian BERT base cased](https://huggingface.co/dbmdz/bert-base-italian-cased) fine-tuned on [italian SQuAD](https://github.com/crux82/squad-it) for **Q&A** downstream task.

## Details of Italian BERT

The source data for the Italian BERT model consists of a recent Wikipedia dump and various texts from the OPUS corpora collection. The final training corpus has a size of 13GB and 2,050,057,573 tokens.

For sentence splitting, we use NLTK (faster compared to spacy). Our cased and uncased models are training with an initial sequence length of 512 subwords for ~2-3M steps.

For the XXL Italian models, we use the same training data from OPUS and extend it with data from the Italian part of the OSCAR corpus. Thus, the final training corpus has a size of 81GB and 13,138,379,147 tokens.
More in its official [model card](https://huggingface.co/dbmdz/bert-base-italian-cased)

Created by [Stefan](https://huggingface.co/stefan-it) at [MDZ](https://huggingface.co/dbmdz)

## Details of the downstream task (Q&A) - Dataset 📚 🧐 ❓

[Italian SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) is derived from the SQuAD dataset and it is obtained through semi-automatic translation of the SQuAD dataset
into Italian. It represents a large-scale dataset for open question answering processes on factoid questions in Italian.
**The dataset contains more than 60,000 question/answer pairs derived from the original English dataset.** The dataset is split into training and test sets to support the replicability of the benchmarking of QA systems:

- `SQuAD_it-train.json`: it contains training examples derived from the original SQuAD 1.1 trainig material.
- `SQuAD_it-test.json`: it contains test/benchmarking examples derived from the origial SQuAD 1.1 development material.

More details about SQuAD-it can be found in [Croce et al. 2018]. The original paper can be found at this [link](https://link.springer.com/chapter/10.1007/978-3-030-03840-3_29).

## Model training 🏋️‍

The model was trained on a Tesla P100 GPU and 25GB of RAM.
The script for fine tuning can be found [here](https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_squad.py)

## Results 📝

| Metric | # Value   |
| ------ | --------- |
| **EM** | **62.51** |
| **F1** | **74.16** |

### Raw metrics

```json
{
  "exact": 62.5180707057432,
  "f1": 74.16038329042492,
  "total": 7609,
  "HasAns_exact": 62.5180707057432,
  "HasAns_f1": 74.16038329042492,
  "HasAns_total": 7609,
  "best_exact": 62.5180707057432,
  "best_exact_thresh": 0.0,
  "best_f1": 74.16038329042492,
  "best_f1_thresh": 0.0
}
```

## Comparison ⚖️

| Model                                                                                                                            | EM        | F1 score  |
| -------------------------------------------------------------------------------------------------------------------------------- | --------- | --------- |
| [DrQA-it trained on SQuAD-it ](https://github.com/crux82/squad-it/blob/master/README.md#evaluating-a-neural-model-over-squad-it) | 56.1      | 65.9      |
| This one                                                                                                                         | **62.51** | **74.16** |

## Model in action 🚀

Fast usage with **pipelines** 🧪

```python
from transformers import pipeline

nlp_qa = pipeline(
    'question-answering',
    model='mrm8488/bert-italian-finedtuned-squadv1-it-alfa',
    tokenizer='mrm8488/bert-italian-finedtuned-squadv1-it-alfa'
)

nlp_qa(
    {
        'question': 'Per quale lingua stai lavorando?',
        'context': 'Manuel Romero è colaborando attivamente con HF / trasformatori per il trader del poder de las últimas ' +
       'técnicas di procesamiento de lenguaje natural al idioma español'
    }
)

# Output: {'answer': 'español', 'end': 174, 'score': 0.9925341537498156, 'start': 168}
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain

Dataset citation

<details>
@InProceedings{10.1007/978-3-030-03840-3_29,
	author="Croce, Danilo and Zelenanska, Alexandra and Basili, Roberto",
	editor="Ghidini, Chiara and Magnini, Bernardo and Passerini, Andrea and Traverso, Paolo",
	title="Neural Learning for Question Answering in Italian",
	booktitle="AI*IA 2018 -- Advances in Artificial Intelligence",
	year="2018",
	publisher="Springer International Publishing",
	address="Cham",
	pages="389--402",
	isbn="978-3-030-03840-3"
}
</detail>
