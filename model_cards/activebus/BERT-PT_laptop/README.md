# ReviewBERT

BERT (post-)trained from review corpus to understand sentiment, options and various e-commence aspects.  

`BERT-DK_laptop` is trained from 100MB laptop corpus under `Electronics/Computers & Accessories/Laptops`. 
`BERT-PT_*` addtionally uses SQuAD 1.1.  

## Model Description

The original model is from `BERT-base-uncased` trained from Wikipedia+BookCorpus.  
Models are post-trained from [Amazon Dataset](http://jmcauley.ucsd.edu/data/amazon/) and [Yelp Dataset](https://www.yelp.com/dataset/challenge/).  


## Instructions
Loading the post-trained weights are as simple as, e.g., 

```python
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("activebus/BERT-PT_laptop")
model = AutoModel.from_pretrained("activebus/BERT-PT_laptop")

```

## Evaluation Results

Check our [NAACL paper](https://www.aclweb.org/anthology/N19-1242.pdf) 


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
