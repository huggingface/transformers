# Retrieval-Augmented Generation (RAG) Example

This example demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline for educational purposes. It illustrates the core concepts of RAG using beginner-friendly code and clear explanations.

## Overview

The example implements a minimal RAG system that:
1. Takes a small corpus of text documents
2. Creates dense vector embeddings for each document using a sentence transformer model
3. Stores these embeddings in a FAISS index for efficient similarity search
4. Given a user query, retrieves the most relevant documents from the index
5. Uses a text generation model to create an answer based on the retrieved context and the query

## Concepts Covered

- **Embeddings**: Converting text into numerical vectors that capture semantic meaning
- **Vector Search**: Using FAISS to find similar documents via embedding similarity
- **Retrieval**: Getting relevant context documents for a given question
- **Generation**: Using a language model to produce answers conditioned on retrieved context
- **RAG Pipeline**: Combining retrieval and generation to enhance factual accuracy

## Installation

To run this example, you need to install the required packages:

```bash
pip install -r requirements.txt
```

Note: This example assumes you have already installed the 🤗 Transformers library from source as per the [repository instructions](https://github.com/huggingface/transformers/tree/main/examples#important-note).

## Running the Example

```bash
python run_rag.py
```

The script will:
1. Download the required models (sentence-transformers/all-MiniLM-L6-v2 and distilgpt2) if not already cached
2. Process the built-in corpus
3. Demonstrate the RAG pipeline with two example questions

## Example Output

```
Step 1: Corpus defined with 8 documents

1. Machine learning is a branch of artificial intelligence.
2. Deep learning is a subset of machine learning that uses neural networks.
3. Natural language processing (NLP) focuses on the interaction between computers and human language.
4. Transformer models have revolutionized NLP tasks in recent years.
5. Retrieval-Augmented Generation combines information retrieval with text generation.
6. FAISS is a library for efficient similarity search and clustering of dense vectors.
7. Sentence transformers convert text into dense vector representations.
8. The process of retrieving relevant documents given a query is called information retrieval.

============================================================

Step 2: Loaded sentence transformer model

Step 3: Encoded corpus into embeddings of shape (8, 384)

============================================================

Step 4: Built FAISS index with 8 vectors

============================================================

Step 5: Initialized text generation pipeline

============================================================

Processing question: What is machine learning?

Retrieved top 2 relevant documents:

1. Distance=0.4089
   Machine learning is a branch of artificial intelligence.

2. Distance=0.5327
   FAISS is a library for efficient similarity search and clustering of dense vectors.

Generating answer...


Running RAG pipeline demonstration
============================================================

Processing question: What is machine learning?

Retrieved top 2 relevant documents:

1. Distance=0.4089
   Machine learning is a branch of artificial intelligence.

2. Distance=0.5327
   FAISS is a library for efficient similarity search and clustering of dense vectors.

Generating answer...


Final Result
------------------------------------------------------------
Question: What is machine learning?
Answer: Machine learning is a branch of artificial intelligence that enables systems to learn from data and improve over time without being explicitly programmed.
============================================================

Processing question: How do transformer models relate to NLP?

Retrieved top 2 relevant documents:

1. Distance=0.3571
   Transformer models have revolutionized NLP tasks in recent years.

2. Distance=0.4823
   Natural language processing (NLP) focuses on the interaction between computers and human language.

Generating answer...


Final Result
------------------------------------------------------------
Question: How do transformer models relate to NLP?
Answer: Transformer models have revolutionized NLP tasks by enabling more effective processing of sequential data through self-attention mechanisms.
============================================================
```

*Note: Actual output may vary slightly due to model generation randomness.*

## Customization

You can modify the following parts of the example:
- **Corpus**: Edit the `corpus` list in `run_rag.py` to use your own documents
- **Embedding model**: Change the `SentenceTransformer` model name (currently `sentence-transformers/all-MiniLM-L6-v2`)
- **Generation model**: Change the `pipeline` model name (currently `distilgpt2`) and adjust generation parameters
- **Number of retrieved documents**: Modify the `k` parameter in the `answer_question` function call
- **Prompt format**: Adjust the prompt construction in the `answer_question` function

## Notes and Limitations

- **Educational Focus**: This example prioritizes clarity and simplicity over production-level performance
- **Small Corpus**: Uses only 8 example documents; real RAG systems typically use much larger document collections
- **Basic Indexing**: Uses FAISS's flat L2 index for exact search; production systems might use approximate indices for scalability
- **Simple Generation**: Uses a basic prompt format; advanced RAG systems may employ more sophisticated prompting strategies
- **Model Sizes**: Uses small models (all-MiniLM-L6-v2 and distilgpt2) for faster demonstration; larger models may yield better results
- **No Training**: Demonstrates inference only; does not include any fine-tuning or training components

This example is intended as a learning tool to understand the fundamental concepts of Retrieval-Augmented Generation. For production applications, consider more advanced implementations with larger datasets, optimized indexing, and specialized RAG architectures.