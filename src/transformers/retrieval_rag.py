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
    def __init__(self, *args, **kwargs):
        pass

    def get_top_docs(self, query_vectors: np.array, n_docs: int = 5):
        raise NotImplementedError

    def init_index(self):
        raise NotImplementedError


class LegacyIndex(Index):
    def __init__(self, vector_size, index_path, passages_path, *args, **kwargs):
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
    def __init__(
        self, dataset, dataset_split, index_name, index_path,
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


class RAGRetriever(object):
    def __init__(self, config):
        super().__init__()
        assert (
            config.retriever_type == "hf_retriever" or config.retriever_type == "mpi_retriever"
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

        # initialize retriever only on the master worker
        if not dist.is_initialized() or self._is_master():
            logger.info("dist not initialized / master")
            self.retriever.init_index()

    def _is_master(self):
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

    def _master_retrieve(self, query_vectors):
        query_vectors_batched = self._chunk_tensor(query_vectors, self.batch_size)
        ids_batched = []
        vectors_batched = []
        for query_vectors in query_vectors_batched:
            start_time = time.time()
            ids, vectors = self.retriever.get_top_docs(query_vectors.numpy(), self.n_docs)
            logger.info(
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

    def retrieve(self, query_vectors, **kwargs):
        # non-ddp initialization (init_retiever() is called at ddp initialization, if no ddp, then it's never called,
        # so it has to be initalized separately.
        if not dist.is_initialized() and self.retriever.index is None:
            logger.info("Initializing index at first query")
            self.retriever.init_index()

        query_vectors_detached = query_vectors.cpu().detach().to(torch.float32)

        # single GPU training
        if not dist.is_initialized():
            doc_ids, doc_vectors = self._master_retrieve(query_vectors_detached)
            return self._finalize_retrieval(query_vectors, doc_ids, doc_vectors)

        # distributed training
        world_size = dist.get_world_size(group=self.process_group)
        logger.info("world_size", world_size)

        # gather logic
        gather_list = None
        if self._is_master():
            gather_list = [torch.empty(query_vectors.shape, dtype=torch.float32) for _ in range(world_size)]
        dist.gather(query_vectors_detached, dst=0, gather_list=gather_list, group=self.process_group)

        # scatter logic
        n_queries = query_vectors.shape[0]
        scatter_ids = []
        scatter_vectors = []
        if self._is_master():
            assert len(gather_list) == world_size
            ids, vectors = self._master_retrieve(torch.cat(gather_list))
            scatter_ids = self._chunk_tensor(ids, n_queries)
            scatter_vectors = self._chunk_tensor(vectors, n_queries)
        doc_ids = self._scattered(scatter_ids, [n_queries, self.n_docs], target_type=torch.int64)
        doc_vectors = self._scattered(scatter_vectors, [n_queries, self.n_docs, query_vectors.shape[1]])

        return self._finalize_retrieval(query_vectors, doc_ids, doc_vectors)
