## ParsBERT: Transformer-based Model for Persian Language Understanding

ParsBERT is a monolingual language model based on Google’s BERT architecture with the same configurations as BERT-Base. 

Paper presenting ParsBERT: [arXiv:2005.12515](https://arxiv.org/abs/2005.12515)

All the models (downstream tasks) are uncased and trained with whole word masking. (coming soon stay tuned)


---

## Introduction

This model is pre-trained on a large Persian corpus with various writing styles from numerous subjects (e.g., scientific, novels, news) with more than 2M documents. A large subset of this corpus was crawled manually.

As a part of ParsBERT methodology, an extensive pre-processing combining POS tagging and WordPiece segmentation was carried out to bring the corpus into a proper format. This process produces more than 40M true sentences. 


## Evaluation

ParsBERT is evaluated on three NLP downstream tasks: Sentiment Analysis (SA), Text Classification, and Named Entity Recognition (NER). For this matter and due to insufficient resources, two large datasets for SA and two for text classification were manually composed, which are available for public use and benchmarking. ParsBERT outperformed all other language models, including multilingual BERT and other hybrid deep learning models for all tasks, improving the state-of-the-art performance in Persian language modeling.

## Results

The following table summarizes the F1 score obtained by ParsBERT as compared to other models and architectures.



### Sentiment Analysis (SA) task

|           Dataset          |  ParsBERT | mBERT | DeepSentiPers |
|:--------------------------:|:---------:|:-----:|:-------------:|
|   Digikala User Comments   |   81.74*  | 80.74 |       -       |
|   SnappFood User Comments  |   88.12*  | 87.87 |       -       |
|   SentiPers (Multi Class)  |   71.11*  |   -   |     69.33     |
|  SentiPers (Binary Class)  |   92.13*  |   -   |     91.98     |



### Text Classification (TC) task

|      Dataset      | ParsBERT | mBERT |
|:-----------------:|:--------:|:-----:|
| Digikala Magazine |   93.59* | 90.72 |
|    Persian News   |   97.19* | 95.79 |


### Named Entity Recognition (NER) task

| Dataset | ParsBERT |  mBERT   | MorphoBERT |  Beheshti-NER  |  LSTM-CRF  |  Rule-Based CRF  |  BiLSTM-CRF  |
|:-------:|:--------:|:--------:|:----------:|:--------------:|:----------:|:----------------:|:------------:|
|  PEYMA  |   93.10* |   86.64  |      -     |      90.59     |      -     |       84.00      |       -      |
|  ARMAN  |   98.79* |   95.89  |    89.9    |      84.03     |    86.55   |         -        |     77.45    |


**If you tested ParsBERT on a public dataset and you want to add your results to the table above, open a pull request or contact us. Also make sure to have your code available online so we can add it as a reference**

## How to use

### TensorFlow 2.0

```python
from transformers import AutoConfig, AutoTokenizer, TFAutoModel

config = AutoConfig.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")

text = "ما در هوشواره معتقدیم با انتقال صحیح دانش و آگاهی، همه افراد می‌توانند از ابزارهای هوشمند استفاده کنند. شعار ما هوش مصنوعی برای همه است."
tokenizer.tokenize(text)

>>> ['ما', 'در', 'هوش', '##واره', 'معتقدیم', 'با', 'انتقال', 'صحیح', 'دانش', 'و', 'اگاهی', '،', 'همه', 'افراد', 'میتوانند', 'از', 'ابزارهای', 'هوشمند', 'استفاده', 'کنند', '.', 'شعار', 'ما', 'هوش', 'مصنوعی', 'برای', 'همه', 'است', '.']

```

### Pytorch

```python
from transformers import AutoConfig, AutoTokenizer, AutoModel

config = AutoConfig.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
```


## NLP Tasks Tutorial 

Coming soon stay tuned


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


## Releases

### Release v0.1 (May 27, 2019)
This is the first version of our ParsBERT based on BERT<sub>BASE</sub>
