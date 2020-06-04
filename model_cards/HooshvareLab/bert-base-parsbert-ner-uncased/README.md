## ParsBERT: Transformer-based Model for Persian Language Understanding

ParsBERT is a monolingual language model based on Google’s BERT architecture with the same configurations as BERT-Base. 

Paper presenting ParsBERT: [arXiv:2005.12515](https://arxiv.org/abs/2005.12515)

All the models (downstream tasks) are uncased and trained with whole word masking. (coming soon stay tuned)


## Persian NER [ARMAN, PEYMA, ARMAN+PEYMA]

This task aims to extract named entities in the text, such as names and label with appropriate `NER` classes such as locations, organizations, etc. The datasets used for this task contain sentences that are marked with `IOB` format. In this format, tokens that are not part of an entity are tagged as `”O”` the `”B”`tag corresponds to the first word of an object, and the `”I”` tag corresponds to the rest of the terms of the same entity. Both `”B”` and `”I”` tags are followed by a hyphen (or underscore), followed by the entity category. Therefore, the NER task is a multi-class token classification problem that labels the tokens upon being fed a raw text. There are two primary datasets used in Persian NER, `ARMAN`, and `PEYMA`. In ParsBERT, we prepared ner for both datasets as well as a combination of both datasets.



### PEYMA

PEYMA dataset includes 7,145 sentences with a total of 302,530 tokens from which 41,148 tokens are tagged with seven different classes.

1. Organization
2. Money
3. Location
4. Date
5. Time
6. Person
7. Percent


|     Label    |   #   |
|:------------:|:-----:|
| Organization | 16964 |
|     Money    |  2037 |
|   Location   |  8782 |
|     Date     |  4259 |
|     Time     |  732  |
|    Person    |  7675 |
|    Percent   |  699  |



**Download**
You can download the dataset from [here](http://nsurl.org/tasks/task-7-named-entity-recognition-ner-for-farsi/)

---

### ARMAN

ARMAN dataset holds 7,682 sentences with 250,015 sentences tagged over six different classes.

1. Organization
2. Location
3. Facility
4. Event
5. Product
6. Person


|     Label    |   #   |
|:------------:|:-----:|
| Organization | 30108 |
|   Location   | 12924 |
|   Facility   |  4458 |
|     Event    |  7557 |
|    Product   |  4389 |
|    Person    | 15645 |



**Download**
You can download the dataset from [here](https://github.com/HaniehP/PersianNER)



## Results

The following table summarizes the F1 score obtained by ParsBERT as compared to other models and architectures.

| Dataset         | ParsBERT | MorphoBERT |  Beheshti-NER  |  LSTM-CRF  |  Rule-Based CRF  |  BiLSTM-CRF  |
|:---------------:|:--------:|:----------:|:--------------:|:----------:|:----------------:|:------------:|
|  ARMAN + PEYMA  |   95.13* |      -     |        -       |      -     |         -        |       -      |
|  PEYMA          |   98.79* |      -     |      90.59     |      -     |       84.00      |       -      |
|  ARMAN          |   93.10* |    89.9    |      84.03     |    86.55   |         -        |     77.45    |


## How to use :hugs:
| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [How to use Pipelines](https://github.com/hooshvare/parsbert-ner/blob/master/persian-ner-pipeline.ipynb)  | Simple and efficient way to use State-of-the-Art models on downstream tasks through transformers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hooshvare/parsbert-ner/blob/master/persian-ner-pipeline.ipynb) |


## Cite 

Please cite the following paper in your publication if you are using [ParsBERT](https://arxiv.org/abs/2005.12515) in your research:

```markdown
@article{ParsBERT,
    title={ParsBERT: Transformer-based Model for Persian Language Understanding},
    author={Mehrdad Farahani, Mohammad Gharachorloo, Marzieh Farahani, Mohammad Manthouri},
    journal={ArXiv},
    year={2020},
    volume={abs/2005.12515}
}
```


## Acknowledgments

We hereby, express our gratitude to the [Tensorflow Research Cloud (TFRC) program](https://tensorflow.org/tfrc) for providing us with the necessary computation resources. We also thank [Hooshvare](https://hooshvare.com) Research Group for facilitating dataset gathering and scraping online text resources.


## Contributors

- Mehrdad Farahani: [Linkedin](https://www.linkedin.com/in/m3hrdadfi/), [Twitter](https://twitter.com/m3hrdadfi), [Github](https://github.com/m3hrdadfi)
- Mohammad Gharachorloo:  [Linkedin](https://www.linkedin.com/in/mohammad-gharachorloo/), [Twitter](https://twitter.com/MGharachorloo), [Github](https://github.com/baarsaam)
- Marzieh Farahani:  [Linkedin](https://www.linkedin.com/in/marziehphi/), [Twitter](https://twitter.com/marziehphi), [Github](https://github.com/marziehphi)
- Mohammad Manthouri:  [Linkedin](https://www.linkedin.com/in/mohammad-manthouri-aka-mansouri-07030766/), [Twitter](https://twitter.com/mmanthouri), [Github](https://github.com/mmanthouri)
- Hooshvare Team:  [Official Website](https://hooshvare.com/), [Linkedin](https://www.linkedin.com/company/hooshvare), [Twitter](https://twitter.com/hooshvare), [Github](https://github.com/hooshvare), [Instagram](https://www.instagram.com/hooshvare/)

+ And a special thanks to Sara Tabrizi for her fantastic poster design. Follow her on: [Linkedin](https://www.linkedin.com/in/sara-tabrizi-64548b79/), [Behance](https://www.behance.net/saratabrizi), [Instagram](https://www.instagram.com/sara_b_tabrizi/)

## Releases

### Release v0.1 (May 29, 2019)
This is the first version of our ParsBERT NER!
