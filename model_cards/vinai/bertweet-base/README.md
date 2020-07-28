# <a name="introduction"></a> BERTweet: A pre-trained language model for English Tweets 

 - BERTweet is the first public large-scale language model pre-trained for English Tweets. BERTweet is trained based on the [RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)  pre-training procedure, using the same model configuration as [BERT-base](https://github.com/google-research/bert). 
 - The corpus used to pre-train BERTweet consists of 850M English Tweets (16B word tokens ~ 80GB), containing 845M Tweets streamed from 01/2012 to 08/2019 and 5M Tweets related to the **COVID-19** pandemic. 
 - BERTweet does better than its competitors RoBERTa-base and [XLM-R-base](https://arxiv.org/abs/1911.02116) and outperforms previous state-of-the-art models on three downstream Tweet NLP tasks of Part-of-speech tagging, Named entity recognition and text classification.

The general architecture and experimental results of BERTweet can be found in our [paper](https://arxiv.org/abs/2005.10200):

    @article{BERTweet,
    title     = {{BERTweet: A pre-trained language model for English Tweets}},
    author    = {Dat Quoc Nguyen, Thanh Vu and Anh Tuan Nguyen},
    journal   = {arXiv preprint},
    volume    = {arXiv:2005.10200},
    year      = {2020}
    }

**Please CITE** our paper when BERTweet is used to help produce published results or incorporated into other software.

For further information or requests, please go to [BERTweet's homepage](https://github.com/VinAIResearch/BERTweet)!

## <a name="transformers"></a> Using BERTweet in HuggingFace's [`transformers`](https://github.com/huggingface/transformers) 

### <a name="install2"></a> Installation 

 -  Python version >= 3.6
 - [PyTorch](http://pytorch.org/) version >= 1.4.0
 - [`transformers`](https://github.com/huggingface/transformers), `fastBPE`, `nltk` and `emoji`: `pip3 install transformers fastBPE nltk emoji`

### <a name="models2"></a> Pre-trained model 

Model | #params | Arch. | Pre-training data
---|---|---|---
`vinai/bertweet-base` | 135M | base | 845M English Tweets (80GB)


### <a name="usage2"></a> Example usage 

```python
import torch
from transformers import BertweetModel, BertweetTokenizer

bertweet = BertweetModel.from_pretrained("vinai/bertweet-base")

# INPUT TWEET IS ALREADY NORMALIZED!
tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base")

line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

input_ids = torch.tensor([tokenizer.encode(line)])

with torch.no_grad():
    features = bertweet(input_ids)  # Models outputs are now tuples
```

## Pre-process raw input Tweets 

Before applying `fastBPE` to the pre-training corpus of 850M English Tweets, we tokenized these  Tweets using `TweetTokenizer` from the NLTK toolkit and used the `emoji` package to translate emotion icons into text strings (here, each icon is referred to as a word token).   We also normalized the Tweets by converting user mentions and web/url links into special tokens `@USER` and `HTTPURL`, respectively. Thus it is recommended to also apply the same pre-processing step for BERTweet-based downstream applications w.r.t. the raw input Tweets.

```python
import torch
from transformers import BertweetTokenizer

# LOAD NORMALIZATION MODE IN THE TOKENIZER IF INPUT TWEET IS RAW!
tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

line = "SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier"

input_ids = torch.tensor([tokenizer.encode(line)])
```

