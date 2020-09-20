## RAG

This is a "base" version of the RAG-Token Model of the the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401.pdf) 
by Patrick Lewis, Ethan Perez, Aleksandara Piktus et al.

## Usage:

```python
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)

input_ids = tokenizer("What is the largest country in the world?", return_tensors="pt").input_ids

generated = model.generate(input_ids=input_ids)
generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)

# => should give [' russia']. Pretty good answer for just having just a dummy dataset.
```
