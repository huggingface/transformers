---
language: en
license: apache-2.0
datasets:
- wiki_dpr
thumbnail: https://huggingface.co/front/thumbnails/facebook.png
---
## RAG

This is a non-finetuned version of the RAG-Token model of the the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401.pdf) 
by Patrick Lewis, Ethan Perez, Aleksandara Piktus et al.

Rag consits of a *question encoder*, *retriever* and a *generator*. The retriever should be a `RagRetriever` instance. The *question encoder* can be any model that can be loaded with `AutoModel` and the *generator* can be any model that can be loaded with `AutoModelForSeq2SeqLM`. 

This model is a non-finetuned RAG-Token model and was created as follows:

```python
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, AutoTokenizer

model = RagTokenForGeneration.from_pretrained_question_encoder_generator("facebook/dpr-question_encoder-single-nq-base", "facebook/bart-large")

question_encoder_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
generator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

tokenizer = RagTokenizer(question_encoder_tokenizer, generator_tokenizer)
model.config.use_dummy_dataset = True
model.config.index_name = "exact"
retriever = RagRetriever(model.config, question_encoder_tokenizer, generator_tokenizer)

model.save_pretrained("./")
tokenizer.save_pretrained("./")
retriever.save_pretrained("./")
```

Note that the model is *uncased* so that all capital input letters are converted to lower-case.

## Usage:

*Note*: the model uses the *dummy* retriever as a default. Better results are obtained by using the full retriever, 
by setting `config.index_name="legacy"` and `config.use_dummy_dataset=False`.
The model can be fine-tuned as follows:

```python
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-token-base")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)

input_dict = tokenizer.prepare_seq2seq_batch("who holds the record in 100m freestyle", "michael phelps", return_tensors="pt") 

outputs = model(input_dict["input_ids"], labels=input_dict["labels"])

loss = outputs.loss

# train on loss
```
