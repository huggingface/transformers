import logging
import random

import ray

from transformers import RagRetriever

logger = logging.getLogger(__name__)

@ray.remote
class RayRetriever(RagRetriever):
    def __init__(self):
        self.initialized = False
        pass

    def init(self, config, question_encoder_tokenizer, generator_tokenizer):
        if not self.initialized:
            super().__init__(config,
                             question_encoder_tokenizer=question_encoder_tokenizer, generator_tokenizer=generator_tokenizer)
            self.initialized = True

    def init_retrieval(self):
        logger.info("initializing retrieval actor")
        self.index.init_index()

    def retrieve(self, question_hidden_states, n_docs):
        doc_ids, retrieved_doc_embeds = self._main_retrieve(
            question_hidden_states, n_docs)
        return retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids)

class RagRayDistributedRetriever(RagRetriever):
    _init_retrieval = False

    def __init__(self, config, question_encoder_tokenizer,
                 generator_tokenizer, retrieval_workers, index=None):
        super().__init__(
            config, question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer, index=index
        )
        self.retrieval_workers = retrieval_workers
        ray.get([worker.init(config, question_encoder_tokenizer,
                             generator_tokenizer) for worker in self.retrieval_workers])

    def init_retrieval(self, num_actors):
        # self.retrieval_workers = [RayRetriever.remote(self.config,
        #                                          self.question_encoder_tokenizer,
        #                                          self.generator_tokenizer)] \
        #                          * num_actors
        ray.get([worker.init_retrieval.remote() for worker in
                 self.retrieval_workers])

    def retrieve(self, question_hidden_states, n_docs):
        assert len(self.retrieval_workers) > 0
        random_worker = random.randint(0, len(self.retrieval_workers)-1)
        return ray.get(self.retrieval_workers[random_worker].retrieve.remote(
            question_hidden_states, n_docs))

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

