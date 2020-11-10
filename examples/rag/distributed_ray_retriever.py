import logging
import random

import ray

from transformers import RagRetriever
from transformers.retrieval_rag import LegacyIndex

logger = logging.getLogger(__name__)

class RayRetriever(RagRetriever):
    _init_retrieval = False

    def __init__(self):
        self.initialized = False
        self.test = False
        # import pdb; pdb.set_trace()
        print("----------------------------------actor created")
        pass

    def init(self, config, question_encoder_tokenizer, generator_tokenizer,
             index):
        if not self.initialized:
            assert not self._init_retrieval
            super(RayRetriever, self).__init__(config,
                             question_encoder_tokenizer=question_encoder_tokenizer, generator_tokenizer=generator_tokenizer, index=index)
            self.initialized = True

    def init_retrieval(self):
        assert not self.test
        print(
            "*******************************************************************************************initializing retrieval actor")
        #assert isinstance(self.index, LegacyIndex)
        self.index.init_index()
        assert self.index is not None
        self.test = True

    def test_f(self):
        return self.test

    def retrieve(self, question_hidden_states, n_docs):
        doc_ids, retrieved_doc_embeds = self._main_retrieve(
            question_hidden_states, n_docs)
        #return retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(
        # doc_ids)
        return doc_ids, retrieved_doc_embeds

class RagRayDistributedRetriever(RagRetriever):
    _init_retrieval = False

    def __init__(self, config, question_encoder_tokenizer,
                 generator_tokenizer, retrieval_workers, index=None):
        #import pdb; pdb.set_trace()
        #assert len(retrieval_workers) == 1
        super().__init__(
            config, question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer, index=index
        )
        self.retrieval_workers = retrieval_workers
        ray.get([worker.init.remote(config, question_encoder_tokenizer,
                             generator_tokenizer, index) for worker in
                 self.retrieval_workers])

    def init_retrieval(self, num_actors):
        # self.retrieval_workers = [RayRetriever.remote(self.config,
        #                                          self.question_encoder_tokenizer,
        #                                          self.generator_tokenizer)] \
        #                          * num_actors
        #import pdb; pdb.set_trace()
        ray.get([worker.init_retrieval.remote() for worker in
                 self.retrieval_workers])

    def retrieve(self, question_hidden_states, n_docs):
        #assert len(self.retrieval_workers) == 2
        random_worker = self.retrieval_workers[random.randint(0,
                                               len(self.retrieval_workers)-1)]
        #assert ray.get(random_worker.test_f.remote())
        # return ray.get(random_worker.retrieve.remote(
        #     question_hidden_states, n_docs))
        doc_ids, retrieved_doc_embeds = ray.get(
            random_worker.retrieve.remote(question_hidden_states, n_docs))
        return retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids)

    @classmethod
    def get_tokenizers(cls, retriever_name_or_path,
                       indexed_dataset=None, **kwargs):
        return super(RagRayDistributedRetriever, cls).get_tokenizers(
            retriever_name_or_path, indexed_dataset, **kwargs)

    @classmethod
    def from_pretrained(cls, retriever_name_or_path, actor_handles,
                        indexed_dataset=None, **kwargs):
        config, question_encoder_tokenizer, generator_tokenizer, index = \
            cls.get_tokenizers(retriever_name_or_path, indexed_dataset,
                               **kwargs)
        return cls(
            config, question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            retrieval_workers=actor_handles,
            index=index
        )

