#!/usr/bin/env python3
"""
A simple Retrieval-Augmented Generation (RAG) example.

This script demonstrates how to build a minimal RAG pipeline using:
- Sentence Transformers for embeddings
- FAISS for vector similarity search
- Hugging Face Transformers for answer generation

The goal is to provide a beginner-friendly introduction to RAG systems.
"""

import faiss
from sentence_transformers import SentenceTransformer

from transformers import pipeline


def main():
    # ------------------------------------------------------------------
    # Step 1: Define a small document corpus
    # ------------------------------------------------------------------

    corpus = [
        "Machine learning is a branch of artificial intelligence.",
        "Deep learning is a subset of machine learning that uses neural networks.",
        "Natural language processing (NLP) focuses on the interaction between computers and human language.",
        "Transformer models have revolutionized NLP tasks in recent years.",
        "Retrieval-Augmented Generation combines information retrieval with text generation.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "Sentence transformers convert text into dense vector representations.",
        "The process of retrieving relevant documents given a query is called information retrieval.",
    ]

    print(f"Step 1: Corpus defined with {len(corpus)} documents\n")

    for i, doc in enumerate(corpus, start=1):
        print(f"{i}. {doc}")

    print("\n" + "=" * 60)

    # ------------------------------------------------------------------
    # Step 2: Load embedding model
    # ------------------------------------------------------------------

    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Step 2: Loaded sentence transformer model")

    # Convert documents into dense vector embeddings
    corpus_embeddings = embedding_model.encode(corpus, show_progress_bar=False)

    print(f"Step 3: Encoded corpus into embeddings of shape {corpus_embeddings.shape}")

    print("\n" + "=" * 60)

    # ------------------------------------------------------------------
    # Step 3: Build FAISS index
    # ------------------------------------------------------------------

    embedding_dimension = corpus_embeddings.shape[1]

    # Using L2 distance for similarity search
    index = faiss.IndexFlatL2(embedding_dimension)

    # Add embeddings into the FAISS index
    index.add(corpus_embeddings)

    print(f"Step 4: Built FAISS index with {index.ntotal} vectors")

    print("\n" + "=" * 60)

    # ------------------------------------------------------------------
    # Step 4: Load text generation pipeline
    # ------------------------------------------------------------------

    generator = pipeline(
        task="text-generation",
        model="distilgpt2",
        max_new_tokens=50,
    )

    print("Step 5: Initialized text generation pipeline")

    print("\n" + "=" * 60)

    # ------------------------------------------------------------------
    # Step 5: Define RAG function
    # ------------------------------------------------------------------

    def answer_question(question: str, k: int = 2) -> str:
        """
        Answer a question using retrieval-augmented generation.

        Args:
            question: Input user question
            k: Number of documents to retrieve

        Returns:
            Generated answer string
        """

        print(f"\nProcessing question: {question}")

        # Encode question into embedding vector
        question_embedding = embedding_model.encode([question], show_progress_bar=False)

        # Search FAISS index
        distances, indices = index.search(question_embedding, k)

        print(f"\nRetrieved top {k} relevant documents:\n")

        retrieved_docs = []

        for rank, (idx, dist) in enumerate(
            zip(indices[0], distances[0]),
            start=1,
        ):
            document = corpus[idx]
            retrieved_docs.append(document)

            print(f"{rank}. Distance={dist:.4f}\n   {document}\n")

        # Combine retrieved documents into context
        context = " ".join(retrieved_docs)

        # Create generation prompt
        prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

        print("Generating answer...\n")

        # Generate response
        outputs = generator(
            prompt,
            num_return_sequences=1,
        )

        generated_text = outputs[0]["generated_text"]

        # Remove prompt from output for cleaner answer
        answer = generated_text.replace(prompt, "").strip()

        return answer

    # ------------------------------------------------------------------
    # Step 6: Run demo questions
    # ------------------------------------------------------------------

    questions = [
        "What is machine learning?",
        "How do transformer models relate to NLP?",
    ]

    print("\nRunning RAG pipeline demonstration")
    print("=" * 60)

    for question in questions:
        answer = answer_question(question)

        print("\nFinal Result")
        print("-" * 60)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("=" * 60)


if __name__ == "__main__":
    main()
