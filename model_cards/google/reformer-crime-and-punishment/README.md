## Reformer Model trained on "Crime and Punishment" 

Crime and Punishment is a novel written by Fyodor Dostoevsky and was translated into English. 

Crime and Punishment training data was taken from `gs://trax-ml/reformer/crime-and-punishment-2554.txt` and contains 
roughly 0.5M tokens. 

The ReformerLM model was trained in flax using colab notebook proposed by authors: https://colab.research.google.com/github/google/trax/blob/master/trax/models/reformer/text_generation.ipynb and the weights were converted to Hugging Face's PyTorch ReformerLM model `ReformerModelWithLMHead`.

The model is a language model that operates on small sub-word units. Text can be generated as follows:

```python
model = ReformerModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment")
tok = ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")
tok.decode(model.generate(tok.encode("A few months later", return_tensors="pt"), do_sample=True,temperature=0.7, max_length=100)[0])

# gives:'A few months later on was more than anything in the flat. 
# “I have already.” “That’s not my notion that he had forgotten him. 
# What does that matter? And why do you mean? It’s only another fellow,” he said as he went out, as though he want'
```
