# shrugging-grace/tweetclassifier

## Model description
This model classifies tweets as either relating to the Covid-19 pandemic or not. 

## Intended uses & limitations
It is intended to be used on tweets commenting on UK politics, in particular those trending with the #PMQs hashtag, as this refers to weekly Prime Ministers' Questions.  

#### How to use
``LABEL_0`` means that the tweet relates to Covid-19

``LABEL_1`` means that the tweet does not relate to Covid-19

## Training data
The model was trained on 1000 tweets (with the "#PMQs'), which were manually labeled by the author. The tweets were collected between May-July 2020. 

### BibTeX entry and citation info

This was based on a pretrained version of BERT. 

@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
