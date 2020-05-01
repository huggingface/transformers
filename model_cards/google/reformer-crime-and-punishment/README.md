## Reformer Model trained on "Crime and Punishment" 
Crime and Punishment text was taken from `gs://trax-ml/reformer/crime-and-punishment-2554.txt`.
Model was trained in flax using colab notebook proposed by authors: https://colab.research.google.com/github/google/trax/blob/master/trax/models/reformer/text_generation.ipynb
Weights were converted to Hugging Face PyTorch `ReformerModelWithLMHead`.

The model is a character level language model. Language from in `crime-and-punishment` style 
can be generated as follows:

```python
model = ReformerModelWithLMHead.from_pretrained("patrickvonplaten/reformer-crime-and-punish")
tok = ReformerTokenizer.from_pretrained("patrickvonplaten/reformer-crime-and-punish")
tok.decode(model.generate(tok.encode("A few months later", return_tensors="pt"), do_sample=True,temperature=0.7, max_length=100)[0])

# gives:'A few months later on was more than anything in the flat. 
# “I have already.” “That’s not my notion that he had forgotten him. 
# What does that matter? And why do you mean? It’s only another fellow,” he said as he went out, as though he want'
```
