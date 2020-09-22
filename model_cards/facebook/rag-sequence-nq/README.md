## RAG

This is the RAG-Sequence Model of the the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401.pdf) 
by Patrick Lewis, Ethan Perez, Aleksandara Piktus et al.

## Usage:

```python

from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="pt")
outputs = model(input_ids=input_dict["input_ids"], labels=input_dict["labels"])

# outputs.loss should give 76.2978

generated = model.generate(input_ids=input_dict["input_ids"])
generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)

# generated_string should give 270,000,000 -> not quite correct the answer, but it also only uses a dummy index
```
