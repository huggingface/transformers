The model can be loaded and used as follows on [this branch](https://github.com/huggingface/transformers/tree/finalize_rag) as follows.


# Load model

```python
from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever

# create Retriever augmented model
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq_new", use_dummy_dataset=True)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq_new", retriever=retriever)

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq_new")

# create input ids and labels
input_ids = tokenizer("who sings does he love me with reba", return_tensors="pt").input_ids

# use labels
labels = tokenizer.generator("Linda Davis", return_tensors="pt").input_ids


# compute loss
outputs = model(input_ids, labels=labels)
```
