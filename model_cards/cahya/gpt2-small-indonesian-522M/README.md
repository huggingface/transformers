---
language: "id"
license: "mit"
datasets:
- Indonesian Wikipedia
widget:
- text: "Pulau Dewata sering dikunjungi"
---

# Indonesian GPT2 small model 

## Model description
It is GPT2-small model pre-trained with indonesian Wikipedia using a causal language modeling (CLM) objective. This 
model is uncased: it does not make a difference between indonesia and Indonesia.

This is one of several other language models that have been pre-trained with indonesian datasets. More detail about 
its usage on downstream tasks (text classification, text generation, etc) is available at [Transformer based Indonesian Language Models](https://github.com/cahya-wirawan/indonesian-language-models/tree/master/Transformers)

## Intended uses & limitations

### How to use
You can use this model directly with a pipeline for text generation. Since the generation relies on some randomness, 
we set a seed for reproducibility:
```python
>>> from transformers import pipeline, set_seed
>>> generator = pipeline('text-generation', model='cahya/gpt2-small-indonesian-522M')
>>> set_seed(42)
>>> generator("Kerajaan Majapahit adalah", max_length=30, num_return_sequences=5, num_beams=10)

[{'generated_text': 'Kerajaan Majapahit adalah sebuah kerajaan yang pernah berdiri di Jawa Timur pada abad ke-14 hingga abad ke-15. Kerajaan ini berdiri pada abad ke-14'}, 
{'generated_text': 'Kerajaan Majapahit adalah sebuah kerajaan yang pernah berdiri di Jawa Timur pada abad ke-14 hingga abad ke-16. Kerajaan ini berdiri pada abad ke-14'}, 
{'generated_text': 'Kerajaan Majapahit adalah sebuah kerajaan yang pernah berdiri di Jawa Timur pada abad ke-14 hingga abad ke-15. Kerajaan ini berdiri pada abad ke-15'}, 
{'generated_text': 'Kerajaan Majapahit adalah sebuah kerajaan yang pernah berdiri di Jawa Timur pada abad ke-14 hingga abad ke-16. Kerajaan ini berdiri pada abad ke-15'}, 
{'generated_text': 'Kerajaan Majapahit adalah sebuah kerajaan yang pernah berdiri di Jawa Timur pada abad ke-14 hingga abad ke-15. Kerajaan ini merupakan kelanjutan dari Kerajaan Majapahit yang'}]

```
Here is how to use this model to get the features of a given text in PyTorch:
```python
from transformers import GPT2Tokenizer, GPT2Model

model_name='cahya/gpt2-small-indonesian-522M'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)
text = "Silakan diganti dengan text apa saja."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```
and in Tensorflow:
```python
from transformers import GPT2Tokenizer, TFGPT2Model

model_name='cahya/gpt2-small-indonesian-522M'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2Model.from_pretrained(model_name)
text = "Silakan diganti dengan text apa saja."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

## Training data

This model was pre-trained with 522MB of indonesian Wikipedia.
The texts are tokenized using a byte-level version of Byte Pair Encoding (BPE) (for unicode characters) and 
a vocabulary size of 52,000. The inputs are sequences of 128 consecutive tokens.
