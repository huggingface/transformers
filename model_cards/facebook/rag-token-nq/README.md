## RAG

This is the RAG-Token Model of the the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401.pdf) 
by Aleksandra Piktus et al.

## Usage:

```python

from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="pt")
outputs = model(input_ids=input_dict["input_ids"], labels=input_dict["labels"])

# outputs.loss should give 76.1230

generated = model.generate(input_ids=input_dict["input_ids"], num_beams=4)
generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)

# generated_string should give 270,000 -> not quite correct the answer, but it also only uses a dummy index
```
