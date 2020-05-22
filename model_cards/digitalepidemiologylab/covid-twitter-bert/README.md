# COVID-Twitter-BERT (CT-BERT)
BERT-large-uncased model, pretrained on a corpus of messages from Twitter about COVID-19

## Overview
This model was trained on 160M tweets collected between January 12 and April 16, 2020 containing at least one of the keywords "wuhan", "ncov", "coronavirus", "covid", or "sars-cov-2". These tweets were filtered and preprocessed to reach a final sample of 22.5M tweets (containing 40.7M sentences and 633M tokens) which were used for training.

This model was evaluated based on downstream classification tasks, but it could be used for any other NLP task which can leverage contextual embeddings. 

In order to achieve best results, make sure to use the same text preprocessing as we did for pretraining. This involves replacing user mentions, urls and emojis. You can find a script on our projects [GitHub repo](https://github.com/digitalepidemiologylab/covid-twitter-bert).

## Example usage
```python
tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert")
model = TFAutoModel.from_pretrained("digitalepidemiologylab/covid-twitter-bert")
```

## References
[1] Martin Müller, Marcel Salaté, Per E Kummervold. "COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter" arXiv preprint arXiv:2005.07503 (2020).
