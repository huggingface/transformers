# <a name="introduction"></a> BERTweet: A pre-trained language model for English Tweets 

 - BERTweet is the first public large-scale language model pre-trained for English Tweets. BERTweet is trained based on the [RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)  pre-training procedure, using the same model configuration as [BERT-base](https://github.com/google-research/bert). 
 - The corpus used to pre-train BERTweet consists of 850M English Tweets (16B word tokens ~ 80GB), containing 845M Tweets streamed from 01/2012 to 08/2019 and 5M Tweets related to the **COVID-19** pandemic. 
 - BERTweet does better than its competitors RoBERTa-base and [XLM-R-base](https://arxiv.org/abs/1911.02116) and outperforms previous state-of-the-art models on three downstream Tweet NLP tasks of Part-of-speech tagging, Named entity recognition and text classification.

The general architecture and experimental results of BERTweet can be found in our [paper](https://arxiv.org/abs/2005.10200):

    @inproceedings{bertweet,
    title     = {{BERTweet: A pre-trained language model for English Tweets}},
    author    = {Dat Quoc Nguyen and Thanh Vu and Anh Tuan Nguyen},
    booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
    year      = {2020}
    }

**Please CITE** our paper when BERTweet is used to help produce published results or is incorporated into other software.

For further information or requests, please go to [BERTweet's homepage](https://github.com/VinAIResearch/BERTweet)!

### <a name="install2"></a> Installation 

 -  Python 3.6+, and PyTorch 1.1.0+ (or TensorFlow 2.0+)
 -  Install `transformers`:
    - `git clone https://github.com/huggingface/transformers.git`
    - `cd transformers`
    - `pip3 install --upgrade .`
 - Install `emoji`: `pip3 install emoji`

### <a name="models2"></a> Pre-trained models 


Model | #params | Arch. | Pre-training data
---|---|---|---
`vinai/bertweet-base` | 135M | base | 845M English Tweets (cased)
`vinai/bertweet-covid19-base-cased` | 135M | base | 23M COVID-19 English Tweets (cased)
`vinai/bertweet-covid19-base-uncased` | 135M | base | 23M COVID-19 English Tweets (uncased)

Two pre-trained models `vinai/bertweet-covid19-base-cased` and `vinai/bertweet-covid19-base-uncased` are resulted by further pre-training the pre-trained model `vinai/bertweet-base` on a  corpus of 23M COVID-19 English Tweets for 40 epochs.  

### <a name="usage2"></a> Example usage 


```python
import torch
from transformers import AutoModel, AutoTokenizer 

bertweet = AutoModel.from_pretrained("vinai/bertweet-covid19-base-cased")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-covid19-base-cased")

# INPUT TWEET IS ALREADY NORMALIZED!
line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

input_ids = torch.tensor([tokenizer.encode(line)])

with torch.no_grad():
    features = bertweet(input_ids)  # Models outputs are now tuples
    
## With TensorFlow 2.0+:
# from transformers import TFAutoModel
# bertweet = TFAutoModel.from_pretrained("vinai/bertweet-covid19-base-cased")
```

### <a name="preprocess"></a> Normalize raw input Tweets 

Before applying `fastBPE` to the pre-training corpus of 850M English Tweets, we tokenized these  Tweets using `TweetTokenizer` from the NLTK toolkit and used the `emoji` package to translate emotion icons into text strings (here, each icon is referred to as a word token).   We also normalized the Tweets by converting user mentions and web/url links into special tokens `@USER` and `HTTPURL`, respectively. Thus it is recommended to also apply the same pre-processing step for BERTweet-based downstream applications w.r.t. the raw input Tweets. BERTweet provides this pre-processing step by enabling the `normalization` argument. 

```python
import torch
from transformers import AutoTokenizer

# Load the AutoTokenizer with a normalization mode if the input Tweet is raw
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-covid19-base-cased", normalization=True)

# from transformers import BertweetTokenizer
# tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-covid19-base-cased", normalization=True)

line = "SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier"

input_ids = torch.tensor([tokenizer.encode(line)])
```
