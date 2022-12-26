# coding=utf-8
# Copyright 2020, The ATLAS Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ATLAS Retriever model implementation."""

import os
import pickle
import time
from typing import Iterable, List, Optional, Tuple
from functools import reduce

import numpy as np

from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends
from .configuration_atlas import AtlasConfig
from .tokenization_atlas import AtlasTokenizer


if is_datasets_available():
    from datasets import Dataset, load_dataset, load_from_disk

if is_faiss_available():
    import faiss


logger = logging.get_logger(__name__)


LEGACY_INDEX_PATH = "https://storage.googleapis.com/huggingface-nlp/datasets/wiki_dpr/"


class Index:
    """
    A base class for the Indices encapsulated by the [`AtlasRetriever`].
    """

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        """
        Returns a list of dictionaries, containing titles and text of the retrieved documents.

        Args:
            doc_ids (`np.ndarray` of shape `(batch_size, n_docs)`):
                A tensor of document indices.
        """
        raise NotImplementedError

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each query in the batch, retrieves `n_docs` documents.

        Args:
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`):
                An array of query vectors.
            n_docs (`int`):
                The number of docs retrieved per query.

        Returns:
            `np.ndarray` of shape `(batch_size, n_docs)`: A tensor of indices of retrieved documents. `np.ndarray` of
            shape `(batch_size, vector_size)`: A tensor of vector representations of retrieved documents.
        """
        raise NotImplementedError

    def is_initialized(self):
        """
        Returns `True` if index is already initialized.
        """
        raise NotImplementedError

    def init_index(self):
        """
        A function responsible for loading the index into memory. Should be called only once per training run of a ATLAS
        model. E.g. if the model is trained on multiple GPUs in a distributed setup, only one of the workers will load
        the index.
        """
        raise NotImplementedError

    def reindex(self):
        """
        A function to reindex all passages with new encoder.
        """
        raise NotImplementedError

class HFIndexBase(Index):
    def __init__(self, vector_size, dataset, index_initialized=False):
        self.vector_size = vector_size
        self.dataset = dataset
        self._index_initialized = index_initialized
        self._check_dataset_format(with_index=index_initialized)
        dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True, dtype="float32")

    def _check_dataset_format(self, with_index: bool):
        if not isinstance(self.dataset, Dataset):
            raise ValueError(f"Dataset should be a datasets.Dataset object, but got {type(self.dataset)}")
        # if len({"title", "text", "embeddings"} - set(self.dataset.column_names)) > 0:
        #     raise ValueError(
        #         "Dataset should be a dataset with the following columns: "
        #         "title (str), text (str) and embeddings (arrays of dimension vector_size), "
        #         f"but got columns {self.dataset.column_names}"
        #     )
        if with_index and "embeddings" not in self.dataset.list_indexes():
            raise ValueError(
                "Missing faiss index in the dataset. Make sure you called `dataset.add_faiss_index` to compute it "
                "or `dataset.load_faiss_index` to load one from the disk."
            )

    def init_index(self):
        raise NotImplementedError()

    def is_initialized(self):
        return self._index_initialized

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        _, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        return np.array(ids), np.array(vectors)  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

class CanonicalHFIndex(HFIndexBase):
    """
    A wrapper around an instance of [`~datasets.Datasets`]. If `index_path` is set to `None`, we load the pre-computed
    index available with the [`~datasets.arrow_dataset.Dataset`], otherwise, we load the index from the indicated path
    on disk.

    Args:
        vector_size (`int`): the dimension of the passages embeddings used by the index
        dataset_name (`str`, optional, defaults to `wiki_dpr`):
            A dataset identifier of the indexed dataset on HuggingFace AWS bucket (list all available datasets and ids
            with `datasets.list_datasets()`).
        dataset_split (`str`, optional, defaults to `train`)
            Which split of the `dataset` to load.
        index_name (`str`, optional, defaults to `train`)
            The index_name of the index associated with the `dataset`. The index loaded from `index_path` will be saved
            under this name.
        index_path (`str`, optional, defaults to `None`)
            The path to the serialized faiss index on disk.
        use_dummy_dataset (`bool`, optional, defaults to `False`):
            If True, use the dummy configuration of the dataset for tests.
    """

    def __init__(
        self,
        vector_size: int,
        dataset_name: str = "wiki_dpr",
        dataset_split: str = "train",
        index_name: Optional[str] = None,
        index_path: Optional[str] = None,
        use_dummy_dataset=False,
    ):
        if int(index_path is None) + int(index_name is None) != 1:
            raise ValueError("Please provide `index_name` or `index_path`.")
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.index_name = index_name
        self.index_path = index_path
        self.use_dummy_dataset = use_dummy_dataset
        logger.info(f"Loading passages from {self.dataset_name}")
        dataset = load_dataset(
            self.dataset_name, with_index=False, split=self.dataset_split, dummy=self.use_dummy_dataset
        )
        super().__init__(vector_size, dataset, index_initialized=False)

    def init_index(self):
        if self.index_path is not None:
            logger.info(f"Loading index from {self.index_path}")
            self.dataset.load_faiss_index("embeddings", file=self.index_path)
        else:
            logger.info(f"Loading index from {self.dataset_name} with index name {self.index_name}")
            self.dataset = load_dataset(
                self.dataset_name,
                with_embeddings=True,
                with_index=True,
                split=self.dataset_split,
                index_name=self.index_name,
                dummy=self.use_dummy_dataset,
            )
            self.dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True)
        self._index_initialized = True


class CustomHFIndex(HFIndexBase):
    """
    A wrapper around an instance of [`~datasets.Datasets`]. The dataset and the index are both loaded from the
    indicated paths on disk.

    Args:
        vector_size (`int`): the dimension of the passages embeddings used by the index
        dataset_path (`str`):
            The path to the serialized dataset on disk. The dataset should have 3 columns: title (str), text (str) and
            embeddings (arrays of dimension vector_size)
        index_path (`str`)
            The path to the serialized faiss index on disk.
    """

    def __init__(self, vector_size: int, dataset, index_path=None):
        super().__init__(vector_size, dataset, index_initialized=index_path is None)
        self.index_path = index_path

    @classmethod
    def load_from_disk(cls, vector_size, dataset_path, index_path):
        logger.info(f"Loading passages from {dataset_path}")
        if dataset_path is None or index_path is None:
            raise ValueError(
                "Please provide `dataset_path` and `index_path` after calling `dataset.save_to_disk(dataset_path)` "
                "and `dataset.get_index('embeddings').save(index_path)`."
            )
        dataset = load_from_disk(dataset_path)
        return cls(vector_size=vector_size, dataset=dataset, index_path=index_path)

    def init_index(self):
        if not self.is_initialized():
            logger.info(f"Loading index from {self.index_path}")
            self.dataset.load_faiss_index("embeddings", file=self.index_path)
            self._index_initialized = True


class AtlasRetriever:
    """
    Retriever used to get documents from vector queries. It retrieves the documents embeddings as well as the documents
    contents, and it formats them to be used with a AtlasModel.

    Args:
        config ([`AtlasConfig`]):
            The configuration of the ATLAS model this Retriever is used with. Contains parameters indicating which
            `Index` to build. You can load your own custom dataset with `config.index_name="custom"` or use a canonical
            one (default) from the datasets library with `config.index_name="wiki_dpr"` for example.
        question_encoder_tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that was used to tokenize the question. It is used to decode the question and then use the
            generator_tokenizer.
        generator_tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for the generator part of the AtlasModel.
        index ([`~models.atlas.retrieval_atlas.Index`], optional, defaults to the one defined by the configuration):
            If specified, use this index instead of the one built using the configuration

    Examples:

    ```python
    >>> # To load the default "wiki_dpr" dataset with 21M passages from wikipedia (index name is 'compressed' or 'exact')
    >>> from transformers import AtlasRetriever

    >>> retriever = AtlasRetriever.from_pretrained(
    ...     "facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed"
    ... )

    >>> # To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/atlas/use_own_knowledge_dataset.py
    >>> from transformers import AtlasRetriever

    >>> dataset = (
    ...     ...
    ... )  # dataset must be a datasets.Datasets object with columns "title", "text" and "embeddings", and it must have a faiss index
    >>> retriever = AtlasRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", indexed_dataset=dataset)

    >>> # To load your own indexed dataset built with the datasets library that was saved on disk. More info in examples/atlas/use_own_knowledge_dataset.py
    >>> from transformers import AtlasRetriever

    >>> dataset_path = "path/to/my/dataset"  # dataset saved via *dataset.save_to_disk(...)*
    >>> index_path = "path/to/my/index.faiss"  # faiss index saved via *dataset.get_index("embeddings").save(...)*
    >>> retriever = AtlasRetriever.from_pretrained(
    ...     "facebook/dpr-ctx_encoder-single-nq-base",
    ...     index_name="custom",
    ...     passages_path=dataset_path,
    ...     index_path=index_path,
    ... )

    >>> # To load the legacy index built originally for Atlas's paper
    >>> from transformers import AtlasRetriever

    >>> retriever = AtlasRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", index_name="legacy")
    ```"""

    def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=None, init_retrieval=True):
        self._init_retrieval = init_retrieval
        requires_backends(self, ["datasets", "faiss"])
        super().__init__()
        self.index = index or self._build_index(config)
        self.generator_tokenizer = generator_tokenizer
        self.question_encoder_tokenizer = question_encoder_tokenizer

        self.n_docs = config.n_docs
        self.batch_size = 1 # TODO: Alex Fix #config.retrieval_batch_size

        self.config = config
        if self._init_retrieval:
            self.init_retrieval()

        self.ctx_encoder_tokenizer = None
        self.return_tokenized_docs = False

    @staticmethod
    def _build_index(config):
        if config.index_name == "custom":
            return CustomHFIndex.load_from_disk(
                vector_size=config.retrieval_vector_size,
                dataset_path=config.passages_path,
                index_path=config.index_path,
            )
        else:
            return CanonicalHFIndex(
                vector_size=config.retrieval_vector_size,
                dataset_name=config.dataset,
                dataset_split=config.dataset_split,
                index_name=config.index_name,
                index_path=config.index_path,
                use_dummy_dataset=config.use_dummy_dataset,
            )

    @classmethod
    def from_pretrained(cls, retriever_name_or_path, indexed_dataset=None, **kwargs):
        requires_backends(cls, ["datasets", "faiss"])
        config = kwargs.pop("config", None) or AtlasConfig.from_pretrained(retriever_name_or_path, **kwargs)
        atlas_tokenizer = AtlasTokenizer.from_pretrained(retriever_name_or_path, config=config)
        question_encoder_tokenizer = atlas_tokenizer.question_encoder
        generator_tokenizer = atlas_tokenizer.generator
        if indexed_dataset is not None:
            config.index_name = "custom"
            index = CustomHFIndex(config.retrieval_vector_size, indexed_dataset)
        else:
            index = cls._build_index(config)
        return cls(
            config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            index=index,
        )

    def save_pretrained(self, save_directory):
        if isinstance(self.index, CustomHFIndex):
            if self.config.index_path is None:
                index_path = os.path.join(save_directory, "hf_dataset_index.faiss")
                self.index.dataset.get_index("embeddings").save(index_path)
                self.config.index_path = index_path
            if self.config.passages_path is None:
                passages_path = os.path.join(save_directory, "hf_dataset")
                # datasets don't support save_to_disk with indexes right now
                faiss_index = self.index.dataset._indexes.pop("embeddings")
                self.index.dataset.save_to_disk(passages_path)
                self.index.dataset._indexes["embeddings"] = faiss_index
                self.config.passages_path = passages_path
        self.config.save_pretrained(save_directory)
        atlas_tokenizer = AtlasTokenizer(
            question_encoder=self.question_encoder_tokenizer,
            generator=self.generator_tokenizer,
        )
        atlas_tokenizer.save_pretrained(save_directory)

    def reindex(self, batch_size: int = 16):
        if self.config.index_name != "custom":
            logger.warning(
                "You are reindexing a pre-indexed dataset. This will overwrite the index with a new one."
                "To reuse this index in the future, you'll need to save it with `retriever.save_pretrained(...)`"
                "and load it with `retriever = AtlasRetriever.from_pretrained(..., index_path=...)`"
            )
        old_index = self.index.dataset.get_index("embeddings")

        device = old_index.device
        string_factory = old_index.string_factory
        metric_type = old_index.metric_type

        def reindex(examples):
            # TODO: make configurable
            tokenized = self.tokenizer(examples['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)

            hidden_states = self.retriever.embed_passages(
                input_ids=tokenized["input_ids"].to(self.device),
                attention_mask=tokenized["attention_mask"].to(self.device)
            )
            examples['embeddings'] = hidden_states.cpu().detach().numpy()
            return examples

        new_dataset = self.dataset.index.map(reindex, batched=True, batch_size=batch_size)
        new_dataset.add_faiss_index("embeddings", device=device, string_factory=string_factory, metric_type=metric_type)

        self.index = CustomHFIndex(self.config.retrieval_vector_size, new_dataset)
        self.index.init_index()

    def init_retrieval(self):
        """
        Retriever initialization function. It loads the index into memory.
        """

        logger.info("initializing retrieval")
        self.index.init_index()

    # TODO: Rewrite
    def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=None):
        r"""
        Postprocessing retrieved `docs` and combining them with `input_strings`.

        Args:
            docs  (`dict`):
                Retrieved documents.
            input_strings (`str`):
                Input strings decoded by `preprocess_query`.
            prefix (`str`):
                Prefix added at the beginning of each input, typically used with T5-based models.

        Return:
            `tuple(tensors)`: a tuple consisting of two elements: contextualized `input_ids` and a compatible
            `attention_mask`.
        """

        def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
            # TODO(Patrick): if we train more ATLAS models, I want to put the input first to take advantage of effortless truncation
            # TODO(piktus): better handling of truncation
            if doc_title.startswith('"'):
                doc_title = doc_title[1:]
            if doc_title.endswith('"'):
                doc_title = doc_title[:-1]
            if prefix is None:
                prefix = ""
            out = (prefix + doc_title + self.config.title_sep + doc_text + self.config.doc_sep + input_string).replace(
                "  ", " "
            )
            return out

        atlas_input_strings = [
            cat_input_and_doc(
                docs[i]["title"][j],
                docs[i]["text"][j],
                input_strings[i],
                prefix,
            )
            for i in range(len(docs))
            for j in range(n_docs)
        ]

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            atlas_input_strings,
            max_length=self.config.max_combined_length,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

    def _chunk_tensor(self, t: Iterable, chunk_size: int) -> List[Iterable]:
        return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]

    def _main_retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray]:
        question_hidden_states_batched = self._chunk_tensor(question_hidden_states, self.batch_size)
        ids_batched = []
        vectors_batched = []
        for question_hidden_states in question_hidden_states_batched:
            start_time = time.time()
            ids, vectors = self.index.get_top_docs(question_hidden_states, n_docs)
            logger.debug(
                f"index search time: {time.time() - start_time} sec, batch size {question_hidden_states.shape}"
            )
            ids_batched.extend(ids)
            vectors_batched.extend(vectors)
        return (
            np.array(ids_batched),
            np.array(vectors_batched),
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]:
        """
        Retrieves documents for specified `question_hidden_states`.

        Args:
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`):
                A batch of query vectors to retrieve with.
            n_docs (`int`):
                The number of docs retrieved per query.

        Return:
            `Tuple[np.ndarray, np.ndarray, List[dict]]`: A tuple with the following objects:

            - **retrieved_doc_embeds** (`np.ndarray` of shape `(batch_size, n_docs, dim)`) -- The retrieval embeddings
              of the retrieved docs per query.
            - **doc_ids** (`np.ndarray` of shape `(batch_size, n_docs)`) -- The ids of the documents in the index
            - **doc_dicts** (`List[dict]`): The `retrieved_doc_embeds` examples per query.
        """

        doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_docs)
        return retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids)

    def set_ctx_encoder_tokenizer(self, ctx_encoder_tokenizer: PreTrainedTokenizer):
        # used in end2end retriever training
        self.ctx_encoder_tokenizer = ctx_encoder_tokenizer
        self.return_tokenized_docs = True

    def __call__(
        self,
        question_input_ids: List[List[int]],
        question_hidden_states: np.ndarray,
        prefix=None,
        n_docs=None,
        return_tensors=None,
    ) -> BatchEncoding:
        """
        Retrieves documents for specified `question_hidden_states`.

        Args:
            question_input_ids: (`List[List[int]]`) batch of input ids
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            prefix: (`str`, *optional*):
                The prefix used by the generator's tokenizer.
            n_docs (`int`, *optional*):
                The number of docs retrieved per query.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.

        Returns: [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **context_input_ids** -- List of token ids to be fed to a model.

              [What are input IDs?](../glossary#input-ids)

            - **context_attention_mask** -- List of indices specifying which tokens should be attended to by the model
            (when `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

              [What are attention masks?](../glossary#attention-mask)

            - **retrieved_doc_embeds** -- List of embeddings of the retrieved documents
            - **doc_ids** -- List of ids of the retrieved documents
        """

        n_docs = n_docs if n_docs is not None else self.n_docs
        prefix = prefix if prefix is not None else self.config.generator.prefix
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)

        input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
        context_input_ids, context_attention_mask = self.postprocess_docs(
            docs, input_strings, prefix, n_docs, return_tensors=return_tensors
        )

        if self.return_tokenized_docs:
            retrieved_doc_text = []
            retrieved_doc_title = []

            for b_idx in range(len(docs)):
                for doc_idx in range(n_docs):
                    retrieved_doc_text.append(docs[b_idx]["text"][doc_idx])
                    retrieved_doc_title.append(docs[b_idx]["title"][doc_idx])

            tokenized_docs = self.ctx_encoder_tokenizer(
                retrieved_doc_title,
                retrieved_doc_text,
                truncation=True,
                padding="longest",
                return_tensors=return_tensors,
            )

            return BatchEncoding(
                {
                    "context_input_ids": context_input_ids,
                    "context_attention_mask": context_attention_mask,
                    "retrieved_doc_embeds": retrieved_doc_embeds,
                    "doc_ids": doc_ids,
                    "tokenized_doc_ids": tokenized_docs["input_ids"],
                    "tokenized_doc_attention_mask": tokenized_docs["attention_mask"],
                },
                tensor_type=return_tensors,
            )

        else:
            return BatchEncoding(
                {
                    "context_input_ids": context_input_ids,
                    "context_attention_mask": context_attention_mask,
                    "retrieved_doc_embeds": retrieved_doc_embeds,
                    "doc_ids": doc_ids,
                },
                tensor_type=return_tensors,
            )


class AtlasRetrieverIndex:
    def __init__(
        self,
        config: AtlasConfig,
        tokenizer: AtlasTokenizer,
        index: Optional[Dataset] = None,
        **kwargs,
    ):
        self.generator_tokenizer = tokenizer.generator
        self.retriever_tokenizer = tokenizer.retriever
        self.config = config
        self.index = index
        if self.index is not None:
            self.set_index(index)

        requires_backends(self, ["datasets", "faiss"])

    def set_index(self, dataset_with_index: Dataset):
        assert isinstance(
            dataset_with_index, Dataset
        ), f"`dataset_with_index` is of type {type(dataset_with_index)}, but should be of type `Dataset`"
        if len({"id", "text", "embeddings"} - set(dataset_with_index.column_names)) > 0:
            raise ValueError(
                "Dataset should be a dataset with the following columns: "
                "id (str), text (str) and embeddings (arrays of dimension vector_size), "
                f"but got columns {dataset_with_index.column_names}"
            )
        if "embeddings" not in dataset_with_index.list_indexes():
            raise ValueError(
                "Missing faiss index in the dataset. Make sure you called `dataset.add_faiss_index` to compute it "
                "or `dataset.load_faiss_index` to load one from the disk."
            )
        self.index = dataset_with_index
        self.index.set_format("numpy", columns=["embeddings"], output_all_columns=True, dtype="float32")
    
    def reindex(self, batch_size: int = 16):
        old_index = self.index.get_index("embeddings")
        device = old_index.device
        string_factory = old_index.string_factory
        metric_type = old_index.metric_type

        def reindex(examples):
            tokenized = self.tokenizer(examples['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)

            hidden_states = self.retriever.embed_passages(
                input_ids=tokenized["input_ids"].to(self.device),
                attention_mask=tokenized["attention_mask"].to(self.device)
            )
            examples['embeddings'] = hidden_states.cpu().detach().numpy()
            return examples

        self.index = self.index.index.map(reindex, batched=True, batch_size=batch_size)
        self.index.add_faiss_index("embeddings", device=device, string_factory=string_factory, metric_type=metric_type)

    def __call__(
        self, 
        retriever_hidden_states,
        generator_input_ids,
        topk: int = 5,
    ):
        _, passage_ids = self.index.search_batch("embeddings", retriever_hidden_states, topk)
        docs = [self.index[[i for i in indices if i >= 0]] for indices in passage_ids]

        queries = self.generator_tokenizer.batch_decode(generator_input_ids, skip_special_tokens=True)

        passages = [[f'{queries[i]} context: {passage}' for passage in doc["text"]] for i, doc in enumerate(docs)]


        def encode_passages(batch, tokenizer, max_length):
            bsz = len(batch)
            n = max([len(example) for example in batch])
            batch = [example + [""] *  (n - len(example)) for example in batch]
            batch = reduce(lambda a, b: a + b, batch)
            tokens = tokenizer(
                batch,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
                truncation=True,
            )
            tokens = {k: v.view(bsz, n, -1) for k, v in tokens.items()}
            
            return tokens

        generator_tokens = encode_passages(passages, self.generator_tokenizer, 512)
        return generator_tokens
