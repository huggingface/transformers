import os
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import faiss
import torch
from datasets import load_dataset

from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)


torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        for passage in split_text(text):
            titles.append(title)
            texts.append(passage)
    return {"title": titles, "text": texts}


def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


def main(
    tmp_dir: str,
    num_proc=2,
    batch_size=16,
    rag_model_name="facebook/rag-sequence-nq",
    dpr_ctx_encoder_model_name="facebook/dpr-ctx_encoder-single-nq-base",
):

    ######################################
    print("Step 1 - Create the dataset")
    ######################################

    # The dataset needed for RAG must have three columns:
    # - title (string): title of the document
    # - text (string): text of a passage of the document
    # - embeddings (array of dimension d): DPR representation of the passage

    # Let's say you have documents in tab-separated csv files with columns "title" and "text"
    csv_path = str(Path(__file__).parent / "test_data" / "my_knowledge_dataset.csv")

    # You can load a Dataset object this way
    dataset = load_dataset("csv", data_files=[csv_path], split="train", delimiter="\t", column_names=["title", "text"])

    # Then split the documents into passages of 100 words
    dataset = dataset.map(split_documents, batched=True, num_proc=num_proc)

    # And compute the embeddings
    # TODO(QL): Use the DPR context encoder trained on the multiset/hybrid dataset instead of single NQ
    ctx_encoder = DPRContextEncoder.from_pretrained(dpr_ctx_encoder_model_name).to(device=device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(dpr_ctx_encoder_model_name)
    dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer), batched=True, batch_size=batch_size
    )

    # And finally save your dataset
    dataset_path = os.path.join(tmp_dir, "my_knowledge_dataset")
    dataset.save_to_disk(dataset_path)

    ######################################
    print("Step 2 - Index the dataset")
    ######################################

    # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
    d = 768  # vectors dimension
    m = 128  # hnsw parameter. Higher is more accurate but takes more time to index (default is 32, 128 should be ok)
    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)

    # And save the index
    index_path = os.path.join(tmp_dir, "my_knowledge_dataset_hnsw_index.faiss")
    dataset.get_index("embeddings").save(index_path)

    ######################################
    print("Step 3 - Load RAG")
    ######################################

    retriever = RagRetriever.from_pretrained(
        rag_model_name, index_name="custom", dataset=dataset_path, index_path=index_path
    )
    model = RagSequenceForGeneration.from_pretrained(rag_model_name, retriever=retriever)
    tokenizer = RagTokenizer.from_pretrained(rag_model_name)

    ######################################
    print("Step 4 - Have fun")
    ######################################

    question = "What does Moses' rod turn into ?"
    input_ids = tokenizer.question_encoder(question, return_tensors="pt")["input_ids"]
    generated = model.generate(input_ids)
    generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    print("Q:", question)
    print("A:", generated_string)


if __name__ == "__main__":
    with TemporaryDirectory() as tmp_dir:
        main(tmp_dir)
