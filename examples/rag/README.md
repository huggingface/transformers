# RAG Examples

The RAG (Retrieval-Augmented Generation) examples have been integrated into the main documentation and model classes.

## 📚 Documentation & Examples

For comprehensive RAG usage examples, see:

- **[RAG Model Documentation](https://huggingface.co/docs/transformers/model_doc/rag)** - Complete API reference with code examples
- **[Chat Templating with RAG](https://huggingface.co/docs/transformers/chat_templating#advanced-retrieval-augmented-generation)** - Advanced RAG usage patterns

## 🚀 Quick Start

```python
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Initialize tokenizer and retriever
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base", 
    dataset="wiki_dpr", 
    index_name="compressed"
)

# Load model
model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-sequence-nq",
    retriever=retriever,
    device_map="auto"
)

# Generate with retrieval
input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", return_tensors="pt")
generated = model.generate(input_ids=input_dict["input_ids"])
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
```

## 📖 Available RAG Models

- `facebook/rag-sequence-nq` - RAG-Sequence for Natural Questions
- `facebook/rag-token-nq` - RAG-Token for Natural Questions  

## 🔍 Model Classes

- [`RagModel`](../../src/transformers/models/rag/modeling_rag.py) - Base RAG model
- [`RagSequenceForGeneration`](../../src/transformers/models/rag/modeling_rag.py) - RAG-Sequence implementation
- [`RagTokenForGeneration`](../../src/transformers/models/rag/modeling_rag.py) - RAG-Token implementation
- [`RagRetriever`](../../src/transformers/models/rag/retrieval_rag.py) - Retrieval component

## 📄 Reference

Based on the paper: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401)