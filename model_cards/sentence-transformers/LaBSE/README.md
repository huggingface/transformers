# LaBSE Pytorch Version 
This is a pytorch port of the tensorflow version of [LaBSE](https://tfhub.dev/google/LaBSE/1).

To get the sentence embeddings, you can  use the following code:
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
model = AutoModel.from_pretrained("sentence-transformers/LaBSE")

sentences = ["Hello World", "Hallo Welt"]

encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded_input, return_dict=True)

embeddings = model_output.pooler_output
embeddings = torch.nn.functional.normalize(embeddings)
print(embeddings)
```


When you have [sentence-transformers](https://www.sbert.net/) installed, you can use the model like this:
```python
from sentence_transformers import SentenceTransformer
sentences = ["Hello World", "Hallo Welt"]

model = SentenceTransformer('LaBSE')
embeddings = model.encode(sentences)
print(embeddings)
```

## Reference:
Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Narveen Ari, Wei Wang. [Language-agnostic BERT Sentence Embedding](https://arxiv.org/abs/2007.01852). July 2020

License: [https://tfhub.dev/google/LaBSE/1](https://tfhub.dev/google/LaBSE/1)
