# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
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
"""RAG Retriever model implementation."""

import os
import pickle
import time

import faiss
import numpy as np
import psutil
import torch
import torch.distributed as dist
from datasets import load_dataset

from .configuration_rag import RagConfig
from .file_utils import cached_path, is_remote_url
from .tokenization_rag import RagTokenizer
from .tokenization_t5 import T5Tokenizer
from .utils import logging


logger = logging.get_logger(__name__)


LEGACY_INDEX_PATH = None  # TODO: add url


class Index(object):
    """
    A base class for the Indices encapsulated by the :class:`~transformers.RagRetriever`.
    """

    def __init__(self, *args, **kwargs):
        pass

    def get_doc_dicts(self, doc_ids):
        """
        Returns a list of dictionaries, containing titles and text of the retrieved documents.

        Args:
            doc_ids (:obj:`torch.Tensor` of shape :obj:`(batch_size, n_docs)`):
                A tensor of document indices.
        """
        pass

    def get_top_docs(self, question_hidden_states, n_docs):
        """
        For each query in the batch, retrieves ``n_docs`` documents.

        Args:
            question_hidden_states (:obj:`np.array` of shape :obj:`(batch_size, vector_size):
                An array of query vectors.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Returns:
            :obj:`torch.Tensor` of shape :obj:`(batch_size, n_docs)`: A tensor of indices of retrieved documents.
            :obj:`torch.Tensor` of shape :obj:`(batch_size, vector_size)`: A tensor of vector representations of retrieved documents.
        """
        raise NotImplementedError

    def is_initialized(self):
        """
        Returns :obj:`True` if index is already initialized.
        """
        raise NotImplementedError

    def init_index(self):
        """
        A function responsible for loading the index into memory. Should be called only once per training run of a RAG model.
        E.g. if the model is trained on multiple GPUs in a distributed setup, only one of the workers will load the index.
        """
        raise NotImplementedError


class LegacyIndex(Index):
    """
    An index which can be deserialized from the files built using https://github.com/facebookresearch/DPR.
    We use default faiss index parameters as specified in that repository.

    Args:
        vector_size (:obj:`int`):
            The dimension of indexed vectors.
        index_path (:obj:`str`):
            Can be either
                - A string with the `identifier name` of a pretrained index compatible with
                  :class:`~transformers.retrieval_rag.LegacyIndex` to load from cache or download,
                  e.g. ``facebook/rag-index``.
                - A path to a `directory` containing index files compatible with
                  :class:`~transformers.retrieval_rag.LegacyIndex`
    """

    INDEX_FILENAME = "hf_bert_base.hnswSQ8_correct_phi_128.c_index"
    PASSAGE_FILENAME = "psgs_w100.tsv.pkl"

    def __init__(self, vector_size, index_path):
        self.index_id_to_db_id = []
        self.index_path = index_path
        self.passages = self._load_passages()
        self.vector_size = vector_size
        self.index = None
        self._index_initialize = False

    def _resolve_path(self, index_path, filename):
        assert os.path.isdir(index_path) or is_remote_url(index_path), "Please specify a valid ``index_path``."
        archive_file = os.path.join(index_path, filename)
        try:
            # Load from URL or cache if already cached
            resolved_archive_file = cached_path(archive_file)
            if resolved_archive_file is None:
                raise EnvironmentError
        except EnvironmentError:
            msg = (
                f"Can't load '{archive_file}'. Make sure that:\n\n"
                f"- '{index_path}' is a correct remote path to a directory containing a file named {filename}"
                f"- or '{index_path}' is the correct path to a directory containing a file named {filename}.\n\n"
            )
            raise EnvironmentError(msg)
        if resolved_archive_file == archive_file:
            logger.info("loading file {}".format(archive_file))
        else:
            logger.info("loading file {} from cache at {}".format(archive_file, resolved_archive_file))
        return resolved_archive_file

    def _load_passages(self):
        logger.info("Loading passages from {}".format(self.index_path))
        passages_path = self._resolve_path(self.index_path, self.PASSAGE_FILENAME)
        with open(passages_path, "rb") as passages_file:
            passages = pickle.load(passages_file)
        return passages

    def _deserialize_index(self):
        logger.info("Loading index from {}".format(self.index_path))
        resolved_index_path = self._resolve_path(self.index_path, self.INDEX_FILENAME + ".index.dpr")
        self.index = faiss.read_index(resolved_index_path)
        resolved_meta_path = self._resolve_path(self.index_path, self.INDEX_FILENAME + ".index_meta.dpr")
        with open(resolved_meta_path, "rb") as metadata_file:
            self.index_id_to_db_id = pickle.load(metadata_file)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def is_initialized(self):
        return self._index_initialize

    def init_index(self):
        index = faiss.IndexHNSWFlat(self.vector_size + 1, 512)
        index.hnsw.efSearch = 128
        index.hnsw.efConstruction = 200
        self.index = index
        self._deserialize_index()
        self._index_initialize = True

    def get_doc_dicts(self, doc_ids):
        doc_list = []
        for doc_ids_i in doc_ids:
            ids = [str(int(doc_id)) for doc_id in doc_ids_i]
            docs = [self.passages[doc_id] for doc_id in ids]
            doc_list.append(docs)
        doc_dicts = []
        for docs in doc_list:
            doc_dict = {}
            doc_dict["title"] = [doc[1] for doc in docs]
            doc_dict["text"] = [doc[0] for doc in docs]
            doc_dicts.append(doc_dict)
        return doc_dicts

    def get_top_docs(self, question_hidden_states: np.array, n_docs: int = 5):
        aux_dim = np.zeros(len(question_hidden_states), dtype="float32").reshape(-1, 1)
        query_nhsw_vectors = np.hstack((question_hidden_states, aux_dim))
        _, docs_ids = self.index.search(query_nhsw_vectors, n_docs)
        vectors = [[self.index.reconstruct(int(doc_id))[:-1] for doc_id in doc_ids] for doc_ids in docs_ids]
        ids = [[int(self.index_id_to_db_id[doc_id]) for doc_id in doc_ids] for doc_ids in docs_ids]
        return torch.tensor(ids), torch.tensor(vectors)


class HFIndex(Index):
    """
    A wrapper around an instance of :class:`~datasets.Datasets`. If ``index_path`` is set to ``None``,
    we load the pre-computed index available with the :class:`~datasets.arrow_dataset.Dataset`, otherwise, we load the index from the indicated path on disk.

    Args:
        dataset (:obj:`str`, optional, defaults to ``wiki_dpr``):
            A datatset identifier of the indexed dataset on HuggingFace AWS bucket (list all available datasets and ids with ``datasets.list_datasets()``).
        dataset_split (:obj:`str`, optional, defaults to ``train``)
            Which split of the ``dataset`` to load.
        index_name (:obj:`str`, optional, defaults to ``train``)
            The index_name of the index associated with the ``dataset``. The index loaded from ``index_path`` will be saved under this name.
        index_path (:obj:`str`, optional, defaults to ``None``)
            The path to the serialized faiss index on disk.
    """

    def __init__(
        self,
        dataset,
        dataset_split,
        index_name,
        index_path,
        use_dummy_dataset,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_split = dataset_split
        self.index_name = index_name
        self.index_path = index_path
        self._index_initialize = False
        self.use_dummy_dataset = use_dummy_dataset

        logger.info("Loading passages from {}".format(self.dataset))
        self.index = load_dataset(
            self.dataset, with_index=False, split=self.dataset_split, dummy=self.use_dummy_dataset
        )

    def is_initialized(self):
        return self._index_initialize

    def init_index(self):
        if self.index_path is not None:
            logger.info("Loading index from {}".format(self.index_path))
            self.index.load_faiss_index(index_name=self.index_name, file=self.index_path)
        else:
            logger.info("Loading index from {}".format(self.dataset + " with index name " + self.index_name))
            self.index = load_dataset(
                self.dataset,
                with_embeddings=True,
                with_index=True,
                split=self.dataset_split,
                index_name=self.index_name,
                dummy=self.use_dummy_dataset,
            )
        self._index_initialize = True

    def get_doc_dicts(self, doc_ids):
        return [self.index[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    def get_top_docs(self, question_hidden_states, n_docs=5):
        _, docs = self.index.get_nearest_examples_batch("embeddings", question_hidden_states, n_docs)
        ids = [[int(i) for i in doc["id"]] for doc in docs]
        vectors = [doc["embeddings"] for doc in docs]
        return torch.tensor(ids), torch.tensor(vectors)


class RagRetriever(object):
    """
    A distributed retriever built on top of the ``torch.distributed`` communication package. During training all workers
    initalize their own instance of the retriever, however, only the main worker loads the index into memory. The index is stored
    in cpu memory. The index will also work well in a non-distributed setup.

    Args:
        config (:class:`~transformers.RagConfig`):
            The configuration of the RAG model this Retriever is used with. Contains parameters indicating which ``Index`` to build.
    """

    def __init__(self, config, generator_tokenizer=None, question_encoder_tokenizer=None):
        super().__init__()
        self.retriever = (
            LegacyIndex(
                config.retrieval_vector_size,
                config.index_path or LEGACY_INDEX_PATH,
            )
            if config.index_name == "legacy"
            else HFIndex(
                config.dataset, config.dataset_split, config.index_name, config.index_path, config.use_dummy_dataset
            )
        )
        # TODO(quentin) use RagTokenizer once the API is defined
        self.generator_tokenizer = generator_tokenizer
        self.question_encoder_tokenizer = question_encoder_tokenizer

        self.process_group = None
        self.n_docs = config.n_docs
        self.batch_size = config.retrieval_batch_size

        if torch.cuda.is_available():
            self.batch_size *= torch.cuda.device_count()

        self.config = config

    @classmethod
    def from_pretrained(cls, retriever_name_or_path, **kwargs):
        config = RagConfig.from_pretrained(retriever_name_or_path, **kwargs)
        rag_tokenizer = RagTokenizer.from_pretrained(retriever_name_or_path, config=config)
        question_encoder_tokenizer = rag_tokenizer.question_encoder
        generator_tokenizer = rag_tokenizer.generator
        return cls(
            config, generator_tokenizer=generator_tokenizer, question_encoder_tokenizer=question_encoder_tokenizer
        )

    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
        rag_tokenizer = RagTokenizer(
            question_encoder_tokenizer=self.question_encoder_tokenizer,
            generator_tokenizer=self.generator_tokenizer,
        )
        rag_tokenizer.save_pretrained(save_directory)

    def init_retrieval(self, distributed_port):
        """
        Retrirever initalization function, needs to be called from the training process. The function sets some common parameters
        and environment variables. On top of that, (only) the main process in the process group loads the index into memory.

        If this functin doesn't get called, we assume we're operating in a non-distributed environment and the index gets loaded
        at first query.

        Args:
            distributed_port (:obj:`int`):
                The port on which the main communication of the training run is carried out. We set the port for retrieval-related
                communication as ``distributed_port + 1``.
        """

        logger.info("initializing retrieval")

        # initializing a separate process group for retrievel as the default
        # nccl backend doesn't support gather/scatter operations while gloo
        # is too slow to replace nccl for the core gpu communication
        if dist.is_initialized():
            logger.info("dist initialized")
            # needs to be set manually
            os.environ["GLOO_SOCKET_IFNAME"] = self._infer_socket_ifname()
            # avoid clash with the NCCL port
            os.environ["MASTER_PORT"] = str(distributed_port + 1)
            self.process_group = dist.new_group(ranks=None, backend="gloo")

        # initialize retriever only on the main worker
        if not dist.is_initialized() or self._is_main():
            logger.info("dist not initialized / main")
            self.retriever.init_index()

        # all processes wait untill the retriever is initialized by the main process
        if dist.is_initialized():
            torch.distributed.barrier(group=self.process_group)

    def preprocess_query(self, input_ids, prefix):
        r"""
        Preprocesses the ``input_id`` by first converting it to string using the ``generator_tokenizer`` and
        then tokenizing it using the ``question_encoder_tokenizer``.

        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Return:
            :obj:`torch.LongTensor`:
                Tokenized input.
            :obj:`str`:
                Decoded input strings.
        """

        input_strings = self.generator_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # handle prefix for T5
        if isinstance(self.generator_tokenizer, T5Tokenizer):
            for i, s in enumerate(input_strings):
                if not s.startswith(prefix):
                    logger.warning("T5 prefix mismatch in {}".format(s))
                if len(input_strings[i]) <= len(prefix):
                    input_strings[i] = ""
                else:
                    input_strings[i] = input_strings[i][len(prefix) :]

        retriever_inputs = self.question_encoder_tokenizer.batch_encode_plus(
            input_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return retriever_inputs["input_ids"].to(input_ids.device), input_strings

    def postprocess_docs(self, docs, input_strings, prefix, n_docs):
        r"""
        Postprocessing retrieved ``docs`` and combining them with ``input_strings``.

        Args:
            doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, n_docs)`):
                Retrieval scores of respective docs - passed for logging.
            docs  (:obj:`dict`):
                Retrieved documents.
            input_strings (:obj:`str`):
                Input strings decoded by ``preprocess_query``.
            prefix (:obj:`str`):
                Prefix added at the beginning of each input, typically used with T5-based models.
            print_docs  (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`True`, documents retrieved during the forward pass will be printed out. Intended for debugging purposes.

        Return:
            :obj:`tuple(tuple(torch.FloatTensor)`:
                a tuple consisting od two elements: contextualized ``input_ids`` and a compatible ``attention_mask``.
        """

        def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
            # TODO(Patrick): if we train more RAG models, I want to put the input first to take advantage of effortless truncation
            # TODO(piktus): better handling of truncation
            if doc_title.startswith('"'):
                doc_title = doc_title[1:]
            if doc_title.endswith('"'):
                doc_title = doc_title[:-1]
            if prefix is None:
                prefix = ""
            # TODO(Patrick, piktus, quention) with current master, T5Tokenizer should add eos token => so `add_eos` is not needed anymore
            #            suffix = self.generator_tokenizer.eos_token if add_eos else ""
            out = (prefix + doc_title + self.config.title_sep + doc_text + self.config.doc_sep + input_string).replace(
                "  ", " "
            )
            return out

        rag_input_strings = [
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
            rag_input_strings,
            max_length=self.config.max_combined_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

    def _is_main(self):
        return dist.get_rank(group=self.process_group) == 0

    def _chunk_tensor(self, t, chunk_size):
        n_chunks = t.shape[0] // chunk_size + int(t.shape[0] % chunk_size > 0)
        return list(torch.chunk(t, n_chunks, dim=0))

    def _scattered(self, scatter_list, target_shape, target_type=torch.float32):
        target_tensor = torch.empty(target_shape, dtype=target_type)
        dist.scatter(target_tensor, src=0, scatter_list=scatter_list, group=self.process_group)
        return target_tensor

    def _infer_socket_ifname(self):
        addrs = psutil.net_if_addrs()
        # a hacky way to deal with varying network interface names
        ifname = next((addr for addr in addrs if addr.startswith("e")), None)
        return ifname

    def _main_retrieve(self, question_hidden_states, n_docs):
        question_hidden_states_batched = self._chunk_tensor(question_hidden_states, self.batch_size)
        ids_batched = []
        vectors_batched = []
        for question_hidden_states in question_hidden_states_batched:
            start_time = time.time()
            ids, vectors = self.retriever.get_top_docs(question_hidden_states.numpy(), n_docs)
            logger.debug(
                "index search time: {} sec, batch size {}".format(
                    time.time() - start_time, question_hidden_states.shape
                )
            )
            ids_batched.append(ids)
            vectors_batched.append(vectors)
        return torch.cat(ids_batched), torch.cat(vectors_batched)

    def retrieve(self, question_hidden_states, n_docs):
        """
        Retrieves documents for specified ``question_hidden_states``. The main process, which has the access to the index stored in memory, gathers queries
        from all the processes in the main training process group, performs the retrieval and scatters back the results.

        Args:
            question_hidden_states (:obj:`torch.Tensor` of shape :obj:`(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Ouput:
            total_scores (:obj:`torch.Tensor` of shape :obj:`(batch_size, n_docs)`
                The retrieval scores of the retrieved docs per query.
            total_examples (:obj:`List[dict]`):
                The retrieved examples per query.
        """

        # non-ddp initialization (init_retrieval() is called at ddp initialization, if no ddp, then it's never called,
        # so it has to be initalized separately.
        if not dist.is_initialized() and not self.retriever.is_initialized():
            logger.info("Initializing index at first query")
            self.retriever.init_index()

        # single GPU training
        if not dist.is_initialized():
            doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_docs)
            return retrieved_doc_embeds, self.retriever.get_doc_dicts(doc_ids)

        # distributed training
        world_size = dist.get_world_size(group=self.process_group)

        # gather logic
        gather_list = None
        if self._is_main():
            gather_list = [torch.empty(question_hidden_states.shape, dtype=torch.float32) for _ in range(world_size)]
        dist.gather(question_hidden_states, dst=0, gather_list=gather_list, group=self.process_group)

        # scatter logic
        n_queries = question_hidden_states.shape[0]
        scatter_ids = []
        scatter_vectors = []
        if self._is_main():
            assert len(gather_list) == world_size
            ids, vectors = self._main_retrieve(torch.cat(gather_list), n_docs)
            scatter_ids = self._chunk_tensor(ids, n_queries)
            scatter_vectors = self._chunk_tensor(vectors, n_queries)
        doc_ids = self._scattered(scatter_ids, [n_queries, n_docs], target_type=torch.int64)
        retrieved_doc_embeds = self._scattered(scatter_vectors, [n_queries, n_docs, question_hidden_states.shape[1]])

        return retrieved_doc_embeds, self.retriever.get_doc_dicts(doc_ids)

    def __call__(self, question_input_ids, question_hidden_states, prefix, n_docs=None):
        # TODO(patrick, quention): would be nice to make retriever framework independent => we could pass `return_tensors='pt'" here
        n_docs = n_docs if n_docs is not None else self.n_docs
        retrieved_doc_embeds, docs = self.retrieve(question_hidden_states, n_docs)

        input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
        context_input_ids, context_attention_mask = self.postprocess_docs(docs, input_strings, prefix, n_docs)

        return context_input_ids, context_attention_mask, retrieved_doc_embeds
