import time

import torch
import torch.distributed as dist
from nlp import load_dataset

from dpr.dense_retriever import DenseHNSWFlatIndexer, DenseRetriever, load_passages

from .modeling_utils import PreTrainedModel
from .tokenization_utils import PreTrainedTokenizer


class Retriever(object):
    def __init__(self, *args, **kwargs):
        pass

    def retrieve(self, query_vectors, ndocs, **kwargs):
        raise NotImplementedError


class MPIRetriever(Retriever):
    def __init__(
        self,
        passages_path,
        vector_size,
        index_path,
        n_docs,
        batch_size,
        use_sampling,
        nucleus_size=-1,
        sampling_temperature=-1,
        distributed_port=-1,
    ):
        super().__init__()
        self.process_group = None
        self.passages = load_passages(passages_path)
        self.n_docs = n_docs
        # replicating the batch sizing logic from Postman
        self.batch_size = batch_size * torch.cuda.device_count()
        self.use_sampling = use_sampling
        self.nucleus_size = nucleus_size
        self.sampling_temperature = sampling_temperature

        # initializing a separate process group for retrievel as the default
        # nccl backend doesn't support gather/scatter operations while gloo
        # is too slow to replace nccl for the core gpu communication
        if dist.is_initialized():
            # needs to be set manually
            os.environ["GLOO_SOCKET_IFNAME"] = self._infer_socket_ifname()
            # avoid clash with the NCCL port
            os.environ["MASTER_PORT"] = str(distributed_port + 1)
            self.process_group = dist.new_group(ranks=None, backend="gloo")

        # initialize retriever only on the master worker
        # TODO(piktus) initialize retrieval per node
        if not dist.is_initialized() or self._is_master():
            index = DenseHNSWFlatIndexer(vector_size)
            index.deserialize_from(index_path)
            self.retriever = DenseRetriever(None, -1, None, index)

    def _is_master(self):
        return dist.get_rank(group=self.process_group) == 0

    def _chunk_tensor(self, t, chunk_size):
        n_chunks = t.shape[0] // chunk_size + t.shape[0] % chunk_size
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

    def _master_retrieve(self, query_tensor, **kwargs):
        query_tensor_npy = query_tensor.numpy()

        # no sampling
        if not self.use_sampling:
            docs_and_scores, vectors = self.retriever.get_top_docs(query_tensor_npy, self.n_docs, with_vectors=True)
            ids, scores = zip(*docs_and_scores)
            ids = torch.tensor([[int(s) for s in ss] for ss in ids])
            scores = torch.bmm(query_tensor.unsqueeze(1), torch.tensor(vectors).transpose(1, 2)).squeeze(1)
            return ids, scores, torch.tensor(vectors)

        # use retrieval sampling
        top_docs_and_scores, vectors = self.retriever.get_top_docs(
            query_tensor_npy, self.nucleus_size, with_vectors=True
        )
        ids, scores = zip(*top_docs_and_scores)
        ids = torch.tensor([[int(s) for s in ss] for ss in ids])
        scores = torch.bmm(query_tensor.unsqueeze(1), torch.tensor(vectors).transpose(1, 2)).squeeze(1)
        n_queries = query_tensor.shape[0]
        probs = torch.softmax(scores / self.sampling_temperature, dim=1)
        sampled_indices = torch.multinomial(probs, self.n_docs, False)
        sampled_scores = scores.new(n_queries, self.n_docs)
        sampled_ids = ids.new(n_queries, self.n_docs)
        sampled_vectors = query_tensor.new(n_queries, self.n_docs, vectors.shape[2])
        for b in range(n_queries):
            sampled_scores[b] = scores[b, sampled_indices[b]]
            sampled_ids[b] = ids[b, sampled_indices[b]]
            sampled_vectors[b] = torch.tensor(vectors[b, sampled_indices[b]])
        return sampled_ids, sampled_scores, sampled_vectors

    def _finalize_retrieval(self, query_vectors, doc_ids, doc_scores, doc_vectors, device):
        doc_ids = doc_ids[:, : self.n_docs]
        doc_vectors = doc_vectors[:, : self.n_docs, :]
        doc_vectors = torch.tensor(doc_vectors)
        doc_scores = torch.bmm(query_vectors.unsqueeze(1), doc_vectors.transpose(1, 2)).squeeze(1)
        doc_list = []
        for i in range(query_vectors.shape[0]):
            ids = [str(int(doc_id)) for doc_id in doc_ids[i]]
            docs = [self.passages[doc_id] for doc_id in ids]
            doc_list.append(docs)
        doc_dicts = []
        for docs in doc_list:
            doc_dict = {}
            doc_dict["title"] = [doc[1] for doc in docs]
            doc_dict["text"] = [doc[0] for doc in docs]
            doc_dicts.append(doc_dict)
        doc_scores = doc_scores

        return doc_scores, doc_dicts

    def retrieve(self, query_vectors, **kwargs):
        device = query_vectors.device
        query_vectors = query_vectors.cpu().detach().to(torch.float32)

        # single GPU training
        if not dist.is_initialized():
            doc_ids, doc_scores, doc_vectors = self._master_retrieve(query_vectors, **kwargs)
            return self._finalize_retrieval(query_vectors, doc_ids, doc_scores, doc_vectors, device)

        # distributed training
        world_size = dist.get_world_size(group=self.process_group)

        # gather logic
        gather_list = None
        if self._is_master():
            gather_list = [torch.empty(query_vectors.shape, dtype=torch.float32) for _ in range(world_size)]
        dist.gather(query_vectors, dst=0, gather_list=gather_list, group=self.process_group)
        # scatter logic
        n_queries = query_vectors.shape[0]
        scatter_ids = []
        scatter_scores = []
        scatter_vectors = []
        if self._is_master():
            assert len(gather_list) == world_size
            # retrieval batching
            query_vectors_batched = self._chunk_tensor(torch.cat(gather_list), self.batch_size)
            batch_doc_ids = []
            batch_doc_scores = []
            batch_doc_vectors = []
            for query_vectors_batch in query_vectors_batched:
                doc_ids, doc_scores, doc_vectors = self._master_retrieve(query_vectors_batch, **kwargs)
                batch_doc_ids.append(doc_ids)
                batch_doc_scores.append(doc_scores)
                batch_doc_vectors.append(doc_vectors)
            scatter_ids = self._chunk_tensor(torch.cat(batch_doc_ids), n_queries)
            scatter_scores = self._chunk_tensor(torch.cat(batch_doc_scores), n_queries)
            scatter_vectors = self._chunk_tensor(torch.cat(batch_doc_vectors), n_queries)
        doc_ids = self._scattered(scatter_ids, [n_queries, self.n_docs], target_type=torch.int64)
        doc_scores = self._scattered(scatter_scores, [n_queries, self.n_docs])
        doc_vectors = self._scattered(scatter_vectors, [n_queries, self.n_docs, query_vectors.shape[1]])

        return self._finalize_retrieval(query_vectors, doc_ids, doc_scores, doc_vectors, device)


class HFRetriever(Retriever):
    """
    Encapsulation of the retrieval index. We may not need a separate class
    for it given how Datasets are evolving but keeping
    it here for now as it may be easier to have it separate when implementing
    multi-node training.

    TODO(piktus): Implement handling of multi-node training.
    """

    def __init__(
        self,
        dataset,
        dataset_name=None,
        dataset_split=None,
        index_name="embeddings",
        doc_encoder: PreTrainedModel = None,
        doc_tokenizer: PreTrainedTokenizer = None,
        uncompressed=False,
        uncompressed_index_path=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.index_name = index_name
        self.doc_encoder = doc_encoder
        self.doc_tokenizer = doc_tokenizer
        assert (self.doc_encoder is None) == (self.doc_tokenizer is None)
        self.device = torch.device("cpu")
        self.uncompressed = uncompressed
        self.uncompressed_index_path = uncompressed_index_path
        self.index = self.build_index() if self.doc_encoder is not None else self.load_index()

    # Build an index from scratch given a dataset. Higly inefficien.
    # Keeping this for now for testing - expects title and text columns
    def build_index(self):
        def _embed_ctx(examples):
            batch_inputs = self.doc_tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=list(zip(examples["title"], examples["text"])),
                return_tensors="pt",
                padding="max_length",
                max_length=None,
                truncation=True,
            )["input_ids"].to(self.device)
            return self.doc_encoder(batch_inputs)[0].cpu().numpy()

        dataset = load_dataset(self.dataset, self.dataset_name, split=self.dataset_split)
        with torch.no_grad():
            dataset_with_embeddings = dataset.map(
                lambda ex: {self.index_name: _embed_ctx(ex)}, batched=True, batch_size=16
            )
        # no device specified - index on CPU
        dataset_with_embeddings.add_faiss_index(column=self.index_name)
        return dataset_with_embeddings

    # Loading a dataset with a pre-computed index.
    def load_index(self):
        # any risk datasets will be silently updated? can I trust it'll remain build by a specific model? add version?
        if self.uncompressed:
            print("Loading uncompressed ")
            dataset = load_dataset(self.dataset, self.dataset_name, with_index=False, split=self.dataset_split)
            # loading uncompressed variant of the index
            dataset.load_faiss_index(
                index_name=self.index_name, file=self.uncompressed_index_path,
            )
            return dataset
        return load_dataset(self.dataset, self.dataset_name, with_index=True, split=self.dataset_split)

    def retrieve(self, question_embs, n_docs=5, **kwargs):
        question_embs = question_embs.detach().cpu().numpy()
        time0 = time.time()
        print("searching {} queries".format(question_embs.shape[0]))
        results = self.index.get_nearest_examples_batch(self.index_name, question_embs, n_docs)
        print(
            "index search time: {} sec for a batch of {} queries.".format(time.time() - time0, question_embs.shape[0])
        )
        return results
