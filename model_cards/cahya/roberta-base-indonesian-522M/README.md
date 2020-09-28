---
language: "id"
license: "mit"
datasets:
- Indonesian Wikipedia
widget:
- text: "Ibu ku sedang bekerja <mask> supermarket."
---

# Indonesian RoBERTa base model (uncased) 

## Model description
It is RoBERTa-base model pre-trained with indonesian Wikipedia using a masked language modeling (MLM) objective. This 
model is uncased: it does not make a difference between indonesia and Indonesia.

This is one of several other language models that have been pre-trained with indonesian datasets. More detail about 
its usage on downstream tasks (text classification, text generation, etc) is available at [Transformer based Indonesian Language Models](https://github.com/cahya-wirawan/indonesian-language-models/tree/master/Transformers)

## Intended uses & limitations

### How to use
You can use this model directly with a pipeline for masked language modeling:
```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='cahya/roberta-base-indonesian-522M')
>>> unmasker("Ibu ku sedang bekerja <mask> supermarket")

```
Here is how to use this model to get the features of a given text in PyTorch:
```python
from transformers import RobertaTokenizer, RobertaModel

model_name='cahya/roberta-base-indonesian-522M'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)
text = "Silakan diganti dengan text apa saja."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```
and in Tensorflow:
```python
from transformers import RobertaTokenizer, TFRobertaModel

model_name='cahya/roberta-base-indonesian-522M'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = TFRobertaModel.from_pretrained(model_name)
text = "Silakan diganti dengan text apa saja."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

## Training data

This model was pre-trained with 522MB of indonesian Wikipedia.
The texts are lowercased and tokenized using WordPiece and a vocabulary size of 32,000. The inputs of the model are 
then of the form:

```<s> Sentence A </s> Sentence B </s>```
