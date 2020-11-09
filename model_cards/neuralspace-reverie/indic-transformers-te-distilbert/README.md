---
language: 
- te
tags:
- MaskedLM
- Telugu
- DistilBERT
- Question-Answering
- Token Classification
- Text Classification
---
# Indic-Transformers Telugu DistilBERT
## Model description
This is a DistilBERT language model pre-trained on ~2 GB of monolingual training corpus. The pre-training data was majorly taken from [OSCAR](https://oscar-corpus.com/).
This model can be fine-tuned on various downstream tasks like text-classification, POS-tagging, question-answering, etc. Embeddings from this model can also be used for feature-based training.
## Intended uses & limitations
#### How to use
```
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('neuralspace-reverie/indic-transformers-te-distilbert')
model = AutoModel.from_pretrained('neuralspace-reverie/indic-transformers-te-distilbert')
text = "మీరు ఎలా ఉన్నారు"
input_ids = tokenizer(text, return_tensors='pt')['input_ids']
out = model(input_ids)[0]
print(out.shape)
# out = [1, 5, 768] 
```
#### Limitations and bias
The original language model has been trained using `PyTorch` and hence the use of `pytorch_model.bin` weights file is recommended. The h5 file for `Tensorflow` has been generated manually by commands suggested [here](https://huggingface.co/transformers/model_sharing.html).
