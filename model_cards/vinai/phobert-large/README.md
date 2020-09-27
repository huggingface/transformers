# <a name="introduction"></a> PhoBERT: Pre-trained language models for Vietnamese 
  
Pre-trained PhoBERT models are the state-of-the-art language models for Vietnamese ([Pho](https://en.wikipedia.org/wiki/Pho), i.e. "Phở", is a popular food in Vietnam):

 - Two PhoBERT versions of "base" and "large" are the first public large-scale monolingual language models pre-trained for Vietnamese. PhoBERT pre-training approach is based on [RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)  which optimizes the [BERT](https://github.com/google-research/bert) pre-training procedure for more robust performance.
 - PhoBERT outperforms previous monolingual and multilingual approaches, obtaining new state-of-the-art performances on four downstream Vietnamese NLP tasks of Part-of-speech tagging, Dependency parsing, Named-entity recognition and Natural language inference.

The general architecture and experimental results of PhoBERT can be found in our EMNLP-2020 Findings [paper](https://arxiv.org/abs/2003.00744):

    @article{phobert,
    title     = {{PhoBERT: Pre-trained language models for Vietnamese}},
    author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},
    journal   = {Findings of EMNLP},
    year      = {2020}
    }

**Please CITE** our paper when PhoBERT is used to help produce published results or is incorporated into other software.

For further information or requests, please go to [PhoBERT's homepage](https://github.com/VinAIResearch/PhoBERT)!

### Installation <a name="install2"></a>
 -  Python 3.6+, and PyTorch 1.1.0+ (or TensorFlow 2.0+)
 -  Install `transformers`:
        - `git clone https://github.com/huggingface/transformers.git`
        - `cd transformers`
        - `pip3 install --upgrade .`

### Pre-trained models <a name="models2"></a>

Model | #params | Arch.  | Pre-training data
---|---|---|---
`vinai/phobert-base` | 135M | base | 20GB  of texts
`vinai/phobert-large` | 370M | large | 20GB  of texts

### Example usage <a name="usage2"></a>

```python
import torch
from transformers import AutoModel, AutoTokenizer

phobert = AutoModel.from_pretrained("vinai/phobert-large")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")

# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
line = "Tôi là sinh_viên trường đại_học Công_nghệ ."

input_ids = torch.tensor([tokenizer.encode(line)])

with torch.no_grad():
    features = phobert(input_ids)  # Models outputs are now tuples

## With TensorFlow 2.0+:
# from transformers import TFAutoModel
# phobert = TFAutoModel.from_pretrained("vinai/phobert-large")
```
