---
language: en
license: apache-2.0
datasets:
- wiki_dpr
---
## RAG

This is the RAG-Token Model of the the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401.pdf) 
by Patrick Lewis, Ethan Perez, Aleksandara Piktus et al.

The model is a *uncased* model, which means that capital letters are simply converted to lower-case letters.

The model consits of a *question_encoder*, *retriever* and a *generator*. The retriever is extracts relevant passages from the *wiki_dpr* `train` datasets, which is linked above.
The question_encoder and retriever are based on `facebook/dpr-question_encoder-single-nq-base` and `facebook/bart-large`, which were jointly finetuned on 
on the *wiki_dpr* QA dataset in an end-to-end fashion.

## Usage:

**Note**: In the usage example below only the *dummy* retriever of *wiki_dpr* is used because the real retriever requires to over 40 GB of RAM.
The model can generate questions to any question as follows:

```python
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

input_dict = tokenizer.prepare_seq2seq_batch("who holds the record in 100m freestyle", return_tensors="pt") 

generated = model.generate(input_ids=input_dict["input_ids"]) 
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0]) 

# should give michael phelps => sounds reasonable
```
