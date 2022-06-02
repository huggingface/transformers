import logging
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import torch
from datasets import Features, Sequence, Value, load_dataset

import faiss
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, HfArgumentParser


logger = logging.getLogger(__name__)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
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
    rag_example_args: "RagExampleArguments",
    processing_args: "ProcessingArguments",
    index_hnsw_args: "IndexHnswArguments",
):

    ######################################
    logger.info("Step 1 - Create the dataset")
    ######################################

    # The dataset needed for RAG must have three columns:
    # - title (string): title of the document
    # - text (string): text of a passage of the document
    # - embeddings (array of dimension d): DPR representation of the passage
    # Let's say you have documents in tab-separated csv files with columns "title" and "text"
    assert os.path.isfile(rag_example_args.csv_path), "Please provide a valid path to a csv file"

    # You can load a Dataset object this way
    dataset = load_dataset(
        "csv", data_files=[rag_example_args.csv_path], split="train", delimiter="\t", column_names=["title", "text"]
    )

    # More info about loading csv files in the documentation: https://huggingface.co/docs/datasets/loading_datasets.html?highlight=csv#csv-files

    # Then split the documents into passages of 100 words
    dataset = dataset.map(split_documents, batched=True, num_proc=processing_args.num_proc)

    # And compute the embeddings
    ctx_encoder = DPRContextEncoder.from_pretrained(rag_example_args.dpr_ctx_encoder_model_name).to(device=device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(rag_example_args.dpr_ctx_encoder_model_name)
    new_features = Features(
        {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))}
    )  # optional, save as float32 instead of float64 to save space
    dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=processing_args.batch_size,
        features=new_features,
    )

    # And finally save your dataset
    passages_path = os.path.join(rag_example_args.output_dir, "my_knowledge_dataset")
    dataset.save_to_disk(passages_path)
    # from datasets import load_from_disk
    # dataset = load_from_disk(passages_path)  # to reload the dataset

    ######################################
    logger.info("Step 2 - Index the dataset")
    ######################################

    # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
    index = faiss.IndexHNSWFlat(index_hnsw_args.d, index_hnsw_args.m, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)

    # And save the index
    index_path = os.path.join(rag_example_args.output_dir, "my_knowledge_dataset_hnsw_index.faiss")
    dataset.get_index("embeddings").save(index_path)
    # dataset.load_faiss_index("embeddings", index_path)  # to reload the index


@dataclass
class RagExampleArguments:
    csv_path: str = field(
        default=str(Path(__file__).parent / "test_run" / "dummy-kb" / "my_knowledge_dataset.csv"),
        metadata={"help": "Path to a tab-separated csv file with columns 'title' and 'text'"},
    )
    question: Optional[str] = field(
        default=None,
        metadata={"help": "Question that is passed as input to RAG. Default is 'What does Moses' rod turn into ?'."},
    )
    rag_model_name: str = field(
        default="facebook/rag-sequence-nq",
        metadata={"help": "The RAG model to use. Either 'facebook/rag-sequence-nq' or 'facebook/rag-token-nq'"},
    )
    dpr_ctx_encoder_model_name: str = field(
        default="facebook/dpr-ctx_encoder-multiset-base",
        metadata={
            "help": (
                "The DPR context encoder model to use. Either 'facebook/dpr-ctx_encoder-single-nq-base' or"
                " 'facebook/dpr-ctx_encoder-multiset-base'"
            )
        },
    )
    output_dir: Optional[str] = field(
        default=str(Path(__file__).parent / "test_run" / "dummy-kb"),
        metadata={"help": "Path to a directory where the dataset passages and the index will be saved"},
    )


@dataclass
class ProcessingArguments:
    num_proc: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use to split the documents into passages. Default is single process."
        },
    )
    batch_size: int = field(
        default=16,
        metadata={
            "help": "The batch size to use when computing the passages embeddings using the DPR context encoder."
        },
    )


@dataclass
class IndexHnswArguments:
    d: int = field(
        default=768,
        metadata={"help": "The dimension of the embeddings to pass to the HNSW Faiss index."},
    )
    m: int = field(
        default=128,
        metadata={
            "help": (
                "The number of bi-directional links created for every new element during the HNSW index construction."
            )
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)

    parser = HfArgumentParser((RagExampleArguments, ProcessingArguments, IndexHnswArguments))
    rag_example_args, processing_args, index_hnsw_args = parser.parse_args_into_dataclasses()
    with TemporaryDirectory() as tmp_dir:
        rag_example_args.output_dir = rag_example_args.output_dir or tmp_dir
        main(rag_example_args, processing_args, index_hnsw_args)
