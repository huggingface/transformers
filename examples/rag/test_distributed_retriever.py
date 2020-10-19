import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from datasets import Dataset

import faiss
from transformers.configuration_bart import BartConfig
from transformers.configuration_dpr import DPRConfig
from transformers.configuration_rag import RagConfig
from transformers.file_utils import is_datasets_available, is_faiss_available, is_psutil_available, is_torch_available
from transformers.retrieval_rag import CustomHFIndex
from transformers.tokenization_bart import BartTokenizer
from transformers.tokenization_bert import VOCAB_FILES_NAMES as DPR_VOCAB_FILES_NAMES
from transformers.tokenization_dpr import DPRQuestionEncoderTokenizer
from transformers.tokenization_roberta import VOCAB_FILES_NAMES as BART_VOCAB_FILES_NAMES


sys.path.append(os.path.join(os.getcwd()))  # noqa: E402 # noqa: E402 # isort:skip

from distributed_retriever import RagPyTorchDistributedRetriever  # noqa: E402 # isort:skip


def require_distributed_retrieval(test_case):
    """
    Decorator marking a test that requires a set of dependencies necessary for pefrorm retrieval with
    :class:`~transformers.RagRetriever`.

    These tests are skipped when respective libraries are not installed.

    """
    if not (is_torch_available() and is_datasets_available() and is_faiss_available() and is_psutil_available()):
        test_case = unittest.skip("test requires PyTorch, Datasets, Faiss, psutil")(test_case)
    return test_case


@require_distributed_retrieval
class RagRetrieverTest(TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        self.retrieval_vector_size = 8

        # DPR tok
        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
        ]
        dpr_tokenizer_path = os.path.join(self.tmpdirname, "dpr_tokenizer")
        os.makedirs(dpr_tokenizer_path, exist_ok=True)
        self.vocab_file = os.path.join(dpr_tokenizer_path, DPR_VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        # BART tok
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        bart_tokenizer_path = os.path.join(self.tmpdirname, "bart_tokenizer")
        os.makedirs(bart_tokenizer_path, exist_ok=True)
        self.vocab_file = os.path.join(bart_tokenizer_path, BART_VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(bart_tokenizer_path, BART_VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_dpr_tokenizer(self) -> DPRQuestionEncoderTokenizer:
        return DPRQuestionEncoderTokenizer.from_pretrained(os.path.join(self.tmpdirname, "dpr_tokenizer"))

    def get_bart_tokenizer(self) -> BartTokenizer:
        return BartTokenizer.from_pretrained(os.path.join(self.tmpdirname, "bart_tokenizer"))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def get_dummy_dataset(self):
        dataset = Dataset.from_dict(
            {
                "id": ["0", "1"],
                "text": ["foo", "bar"],
                "title": ["Foo", "Bar"],
                "embeddings": [np.ones(self.retrieval_vector_size), 2 * np.ones(self.retrieval_vector_size)],
            }
        )
        dataset.add_faiss_index("embeddings", string_factory="Flat", metric_type=faiss.METRIC_INNER_PRODUCT)
        return dataset

    def get_dummy_pytorch_distributed_retriever(
        self, init_retrieval: bool, port=12345
    ) -> RagPyTorchDistributedRetriever:
        dataset = self.get_dummy_dataset()
        config = RagConfig(
            retrieval_vector_size=self.retrieval_vector_size,
            question_encoder=DPRConfig().to_dict(),
            generator=BartConfig().to_dict(),
        )
        with patch("transformers.retrieval_rag.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = dataset
            retriever = RagPyTorchDistributedRetriever(
                config,
                question_encoder_tokenizer=self.get_dpr_tokenizer(),
                generator_tokenizer=self.get_bart_tokenizer(),
            )
            if init_retrieval:
                retriever.init_retrieval(port)
        return retriever

    def get_dummy_custom_hf_index_retriever(self, init_retrieval: bool, from_disk: bool, port=12345):
        dataset = self.get_dummy_dataset()
        config = RagConfig(
            retrieval_vector_size=self.retrieval_vector_size,
            question_encoder=DPRConfig().to_dict(),
            generator=BartConfig().to_dict(),
            index_name="custom",
        )
        if from_disk:
            config.passages_path = os.path.join(self.tmpdirname, "dataset")
            config.index_path = os.path.join(self.tmpdirname, "index.faiss")
            dataset.get_index("embeddings").save(os.path.join(self.tmpdirname, "index.faiss"))
            dataset.drop_index("embeddings")
            dataset.save_to_disk(os.path.join(self.tmpdirname, "dataset"))
            del dataset
            retriever = RagPyTorchDistributedRetriever(
                config,
                question_encoder_tokenizer=self.get_dpr_tokenizer(),
                generator_tokenizer=self.get_bart_tokenizer(),
            )
        else:
            retriever = RagPyTorchDistributedRetriever(
                config,
                question_encoder_tokenizer=self.get_dpr_tokenizer(),
                generator_tokenizer=self.get_bart_tokenizer(),
                index=CustomHFIndex(config.retrieval_vector_size, dataset),
            )
        if init_retrieval:
            retriever.init_retrieval(port)
        return retriever

    def test_pytorch_distributed_retriever_retrieve(self):
        n_docs = 1
        retriever = self.get_dummy_pytorch_distributed_retriever(init_retrieval=True)
        hidden_states = np.array(
            [np.ones(self.retrieval_vector_size), -np.ones(self.retrieval_vector_size)], dtype=np.float32
        )
        retrieved_doc_embeds, doc_ids, doc_dicts = retriever.retrieve(hidden_states, n_docs=n_docs)
        self.assertEqual(retrieved_doc_embeds.shape, (2, n_docs, self.retrieval_vector_size))
        self.assertEqual(len(doc_dicts), 2)
        self.assertEqual(sorted(doc_dicts[0]), ["embeddings", "id", "text", "title"])
        self.assertEqual(len(doc_dicts[0]["id"]), n_docs)
        self.assertEqual(doc_dicts[0]["id"][0], "1")  # max inner product is reached with second doc
        self.assertEqual(doc_dicts[1]["id"][0], "0")  # max inner product is reached with first doc
        self.assertListEqual(doc_ids.tolist(), [[1], [0]])

    def test_custom_hf_index_retriever_retrieve(self):
        n_docs = 1
        retriever = self.get_dummy_custom_hf_index_retriever(init_retrieval=True, from_disk=False)
        hidden_states = np.array(
            [np.ones(self.retrieval_vector_size), -np.ones(self.retrieval_vector_size)], dtype=np.float32
        )
        retrieved_doc_embeds, doc_ids, doc_dicts = retriever.retrieve(hidden_states, n_docs=n_docs)
        self.assertEqual(retrieved_doc_embeds.shape, (2, n_docs, self.retrieval_vector_size))
        self.assertEqual(len(doc_dicts), 2)
        self.assertEqual(sorted(doc_dicts[0]), ["embeddings", "id", "text", "title"])
        self.assertEqual(len(doc_dicts[0]["id"]), n_docs)
        self.assertEqual(doc_dicts[0]["id"][0], "1")  # max inner product is reached with second doc
        self.assertEqual(doc_dicts[1]["id"][0], "0")  # max inner product is reached with first doc
        self.assertListEqual(doc_ids.tolist(), [[1], [0]])

    def test_custom_pytorch_distributed_retriever_retrieve_from_disk(self):
        n_docs = 1
        retriever = self.get_dummy_custom_hf_index_retriever(init_retrieval=True, from_disk=True)
        hidden_states = np.array(
            [np.ones(self.retrieval_vector_size), -np.ones(self.retrieval_vector_size)], dtype=np.float32
        )
        retrieved_doc_embeds, doc_ids, doc_dicts = retriever.retrieve(hidden_states, n_docs=n_docs)
        self.assertEqual(retrieved_doc_embeds.shape, (2, n_docs, self.retrieval_vector_size))
        self.assertEqual(len(doc_dicts), 2)
        self.assertEqual(sorted(doc_dicts[0]), ["embeddings", "id", "text", "title"])
        self.assertEqual(len(doc_dicts[0]["id"]), n_docs)
        self.assertEqual(doc_dicts[0]["id"][0], "1")  # max inner product is reached with second doc
        self.assertEqual(doc_dicts[1]["id"][0], "0")  # max inner product is reached with first doc
        self.assertListEqual(doc_ids.tolist(), [[1], [0]])
