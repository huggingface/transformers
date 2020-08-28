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

import logging
import os
import pickle
import time

import faiss
import numpy as np
import psutil
import torch
import torch.distributed as dist
from nlp import load_dataset


logger = logging.getLogger(__name__)


class Index(object):
    """
    A base class for the Indices encapsulated by the :class:`~transformers.RagRetriever`.
    """

    def __init__(self, *args, **kwargs):
        pass

    def get_top_docs(self, query_vectors, n_docs):
        """
        An index which can be deserialized from the files built using https://github.com/facebookresearch/DPR.
        We use default faiss index parameters as specified in that repository.
        Args:
            query_vectors (:obj:`np.array`):
                An array of query vectors.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.
        """
        raise NotImplementedError

    def init_index(self):
        """
        A function responsible for loading the index to memory. Should be called only once per training run of a RAG model.
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
            The path to the serialized faiss index on disk.
        passages_path: (:obj:`str`):
            A path to text passages on disk, compatible with the faiss index.
    """

    def __init__(self, vector_size, index_path, passages_path):
        self.index_id_to_db_id = []
        with open(passages_path, "rb") as passages_file:
            self.passages = pickle.load(passages_file)
        self.index_path = index_path
        self.vector_size = vector_size
        self.index = None

    def _deserialize_from(self, index_path: str):
        logger.info("Loading index from {}".format(index_path))
        self.index = faiss.read_index(index_path + ".index.dpr")
        with open(index_path + ".index_meta.dpr", "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def init_index(self):
        index = faiss.IndexHNSWFlat(self.vector_size + 1, 512)
        index.hnsw.efSearch = 128
        index.hnsw.efConstruction = 200
        self.index = index
        self._deserialize_from(self.index_path)

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

    def get_top_docs(self, query_vectors: np.array, n_docs: int = 5):
        aux_dim = np.zeros(len(query_vectors), dtype="float32").reshape(-1, 1)
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim))
        _, docs_ids = self.index.search(query_nhsw_vectors, n_docs)
        vectors = [[self.index.reconstruct(int(doc_id))[:-1] for doc_id in doc_ids] for doc_ids in docs_ids]
        ids = [[int(self.index_id_to_db_id[doc_id]) for doc_id in doc_ids] for doc_ids in docs_ids]
        return torch.tensor(ids), torch.tensor(vectors)


class HFIndex(Index):
    """
    An and index build for an instance of :class:`~nlp.Datasets`. If ``index_path`` is set to ``None``,
    we load the pre-computed index available with the :class:`~nlp.arrow_dataset.Dataset`, otherwise, we load the index from the indicated path on disk.

    Args:
        dataset (:obj:`str`, optional, defaults to ``wiki_dpr``):
            A datatset identifier of the indexed dataset on HuggingFace AWS bucket (list all available datasets and ids with ``nlp.list_datasets()``).
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
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_split = dataset_split
        self.index_name = index_name
        self.index_path = index_path
        self.index = None

    def init_index(self):
        if self.index_path is not None:
            self.index = load_dataset(self.dataset, with_index=False, split=self.dataset_split)
            self.index.load_faiss_index(index_name=self.index_name, file=self.index_path)
        else:
            self.index = load_dataset(self.dataset, with_embeddings=True, with_index=True, split=self.dataset_split)

    def get_doc_dicts(self, doc_ids):
        return [self.index[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    def get_top_docs(self, query_vectors, n_docs=5):
        _, docs = self.index.get_nearest_examples_batch(self.index_name, query_vectors, n_docs)
        ids = [[int(i) for i in doc["id"]] for doc in docs]
        vectors = [doc[self.index_name] for doc in docs]
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

    def __init__(self, config):
        super().__init__()
        assert (
            config.retriever_type == "hf_retriever" or config.retriever_type == "legacy_retriever"
        ), "invalid retirever type"

        self.retriever = (
            HFIndex(config.dataset, config.dataset_split, config.index_name, config.index_path)
            if config.retriever_type == "hf_retriever"
            else LegacyIndex(config.retrieval_vector_size, config.index_path, config.passages_path)
        )
        self.process_group = None
        self.n_docs = config.n_docs
        self.batch_size = config.retrieval_batch_size * torch.cuda.device_count()

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

    def _main_retrieve(self, query_vectors):
        query_vectors_batched = self._chunk_tensor(query_vectors, self.batch_size)
        ids_batched = []
        vectors_batched = []
        for query_vectors in query_vectors_batched:
            start_time = time.time()
            ids, vectors = self.retriever.get_top_docs(query_vectors.numpy(), self.n_docs)
            logger.debug(
                "index search time: {} sec, batch size {}".format(time.time() - start_time, query_vectors.shape)
            )
            ids_batched.append(ids)
            vectors_batched.append(vectors)
        return torch.cat(ids_batched), torch.cat(vectors_batched)

    def _finalize_retrieval(self, query_vectors, doc_ids, doc_vectors):
        assert doc_ids.shape[1] == self.n_docs and doc_vectors.shape[1] == self.n_docs
        doc_vectors = doc_vectors.to(query_vectors)
        doc_scores = torch.bmm(query_vectors.unsqueeze(1), doc_vectors.transpose(1, 2)).squeeze(1)
        doc_dicts = self.retriever.get_doc_dicts(doc_ids)
        return doc_scores, doc_dicts

    def retrieve(self, query_vectors, n_docs):
        """
        Retrieves documents for specified ``query_vectors``. The main process, which has the access to the index stored in memory, gathers queries
        from all the processes in the main training process group, performs the retrieval and scatters back the results.

        Args:
            query_vectors (:obj:`torch.Tensor` of shape :obj:`(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Ouput:
            total_scores (:obj:`torch.Tensor` of shape :obj:`(batch_size, vector_size)`
                The retrieval scores of the retrieved examples per query.
            total_examples (:obj:`List[dict]`):
                The retrieved examples per query.
        """

        # non-ddp initialization (init_retiever() is called at ddp initialization, if no ddp, then it's never called,
        # so it has to be initalized separately.
        if not dist.is_initialized() and self.retriever.index is None:
            logger.info("Initializing index at first query")
            self.retriever.init_index()

        query_vectors_detached = query_vectors.cpu().detach().to(torch.float32)

        # single GPU training
        if not dist.is_initialized():
            doc_ids, doc_vectors = self._main_retrieve(query_vectors_detached)
            return self._finalize_retrieval(query_vectors, doc_ids, doc_vectors)

        # distributed training
        world_size = dist.get_world_size(group=self.process_group)

        # gather logic
        gather_list = None
        if self._is_main():
            gather_list = [torch.empty(query_vectors.shape, dtype=torch.float32) for _ in range(world_size)]
        dist.gather(query_vectors_detached, dst=0, gather_list=gather_list, group=self.process_group)

        # scatter logic
        n_queries = query_vectors.shape[0]
        scatter_ids = []
        scatter_vectors = []
        if self._is_main():
            assert len(gather_list) == world_size
            ids, vectors = self._main_retrieve(torch.cat(gather_list))
            scatter_ids = self._chunk_tensor(ids, n_queries)
            scatter_vectors = self._chunk_tensor(vectors, n_queries)
        doc_ids = self._scattered(scatter_ids, [n_queries, self.n_docs], target_type=torch.int64)
        doc_vectors = self._scattered(scatter_vectors, [n_queries, self.n_docs, query_vectors.shape[1]])

        return self._finalize_retrieval(query_vectors, doc_ids, doc_vectors)
