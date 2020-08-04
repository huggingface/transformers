---
language: en
tags:
- exbert
license: apache-2.0
datasets:
- snli
- multi_nli
---

# BERT base model (uncased) for Sentence Embeddings
This is the `bert-base-nli-cls-token` model from the [sentence-transformers](https://github.com/UKPLab/sentence-transformers)-repository. The sentence-transformers repository allows to train and use Transformer models for generating sentence and text embeddings. 
The model is described in  the paper  [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

## Usage (HuggingFace Models Repository)

You can use the model directly from the model repository to compute sentence embeddings. The CLS token of each input represents the sentence embedding:
```python
from transformers import AutoTokenizer, AutoModel
import torch


#Sentences we want sentence embeddings for
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']

#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-cls-token")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-cls-token")

#Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

#Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    sentence_embeddings = model_output[0][:,0] #Take the first token ([CLS]) from each sentence 

print("Sentence embeddings:")
print(sentence_embeddings)
```

## Usage (Sentence-Transformers)
Using this model becomes more convenient when you have [sentence-transformers](https://github.com/UKPLab/sentence-transformers) installed:
```
pip install -U sentence-transformers
```

Then you can use the model like this:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-cls-token')
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)

print("Sentence embeddings:")
print(sentence_embeddings)
```


## Citing & Authors
If you find this model helpful, feel free to cite our publication [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084):
``` 
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
```
