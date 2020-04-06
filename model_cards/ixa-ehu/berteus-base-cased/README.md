---
language:
- basque
---

# BERTeus base cased

This is the Basque language pretrained model presented in [Give your Text Representation Models some Love: the Case for Basque](https://arxiv.org/pdf/2004.00033.pdf). This model has been trained on a Basque corpus comprising Basque crawled news articles from online newspapers and the Basque Wikipedia. The training corpus contains 224.6 million tokens, of which 35 million come from the Wikipedia.

BERTeus has been tested on four different downstream tasks for Basque: part-of-speech (POS) tagging, named entity recognition (NER), sentiment analysis and topic classification; improving the state of the art for all tasks. See summary of results below:


| Downstream task | BERTeus | mBERT | Previous SOTA |
| --------------- | ------- | ------| ------------- |
| Topic Classification	  | **76.77**   | 68.42 | 63.00 	    |
| Sentiment    	  | **78.10**   | 71.02 | 74.02 	    |
| POS   	  | **97.76**   | 96.37 | 96.10 	    |
| NER    	  | **87.06**   | 81.52 | 76.72 	    |


If using this model, please cite the following paper:
```
@inproceedings{agerri2020give,
  title={Give your Text Representation Models some Love: the Case for Basque},
  author={Rodrigo Agerri and I{\~n}aki San Vicente and Jon Ander Campos and Ander Barrena and Xabier Saralegi and Aitor Soroa and Eneko Agirre},
  booktitle={Proceedings of the 12th International Conference on Language Resources and Evaluation},
  year={2020}
}
```
