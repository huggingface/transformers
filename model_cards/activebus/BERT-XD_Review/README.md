# ReviewBERT

BERT (post-)trained from review corpus to understand sentiment, options and various e-commence aspects.  
Please visit https://github.com/howardhsu/BERT-for-RRC-ABSA for details.  

`BERT-XD_Review` is a cross-domain (beyond just `laptop` and `restaurant`) language model, where each example is from a single product / restaurant with the same rating, post-trained (fine-tuned) on a combination of 5-core Amazon reviews and all Yelp data, expected to be 22 G in total. It is trained for 4 epochs on `bert-base-uncased`.
The preprocessing code [here](https://github.com/howardhsu/BERT-for-RRC-ABSA/transformers).

## Model Description

The original model is from `BERT-base-uncased`.  
Models are post-trained from [Amazon Dataset](http://jmcauley.ucsd.edu/data/amazon/) and [Yelp Dataset](https://www.yelp.com/dataset/challenge/).  


## Instructions
Loading the post-trained weights are as simple as, e.g., 

```python
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("activebus/BERT-XD_Review")
model = AutoModel.from_pretrained("activebus/BERT-XD_Review")

```


## Evaluation Results

Check our [NAACL paper](https://www.aclweb.org/anthology/N19-1242.pdf) 
`BERT_Review` is expected to have similar performance on domain-specific tasks (such as aspect extraction) as `BERT-DK`, but much better on general tasks such as aspect sentiment classification (different domains mostly share similar sentiment words).


## Citation
If you find this work useful, please cite as following.
```
@inproceedings{xu_bert2019,
    title = "BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis",
    author = "Xu, Hu and Liu, Bing and Shu, Lei and Yu, Philip S.",
    booktitle = "Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics",
    month = "jun",
    year = "2019",
}
```
