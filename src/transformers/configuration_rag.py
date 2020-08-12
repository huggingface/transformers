# coding=utf-8
# Copyright 2010, XXX authors
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
""" XXX model configuration """


import logging

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

RAG_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # "rag-sequence-default": "/private/home/piktus/rag_huggingface/data/rag-sequence-nq/",
    # "rag-token-default": "/private/home/piktus/rag_huggingface/data/rag-token-nq/",
}


class RagConfig(PretrainedConfig):
    r"""
        :class:`~transformers.RagConfig` is the configuration class to store the configuration of a
        `RagModel`.
        Arguments:
            TBA
    """
    model_type = "xxx"

    def __init__(
        self,
        vocab_size=50264,
        is_encoder_decoder=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
        title_sep=" / ",
        doc_sep=" // ",
        n_docs=5,
        max_combined_length=300,  # max token length of input with context doc prepended
        retriever_type="hf_retriever",
        # hf_retriever default params
        dataset="wiki_dpr",
        dataset_name="psgs_w100_no_embeddings",
        dataset_split="train",
        index_name="embeddings",
        uncompressed=False,
        uncompressed_index_path=None,
        # mpi_retriever default params
        retrieval_vector_size=786,
        retrieval_batch_size=8,
        passages_path=None,
        index_path=None,
        # pre-trained components
        pretrained_context_encoder_name_or_path="facebook/dpr-ctx_encoder-single-nq-base",
        pretrained_context_tokenizer_name_or_path="facebook/dpr-ctx_encoder-single-nq-base",
        pretrained_question_encoder_name_or_path="facebook/dpr-question_encoder-single-nq-base",
        pretrained_generator_tokenizer_name_or_path="facebook/bart-large",
        pretrained_generator_name_or_path="/private/home/piktus/rag_hugginface/data/rag-sequence-nq",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.is_encoder_decoder = is_encoder_decoder
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.title_sep = title_sep
        self.doc_sep = doc_sep
        self.n_docs = n_docs
        self.max_combined_length = max_combined_length

        self.retriever_type = retriever_type

        self.dataset = dataset
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.index_name = index_name

        self.retrieval_vector_size = retrieval_vector_size
        self.retrieval_batch_size = retrieval_batch_size
        self.passages_path = passages_path
        self.index_path = index_path
        self.uncompressed = uncompressed
        self.uncompressed_index_path = uncompressed_index_path

        self.pretrained_context_encoder_name_or_path = pretrained_context_encoder_name_or_path
        self.pretrained_context_tokenizer_name_or_path = pretrained_context_tokenizer_name_or_path
        self.pretrained_question_encoder_name_or_path = pretrained_question_encoder_name_or_path
        self.pretrained_generator_tokenizer_name_or_path = pretrained_generator_tokenizer_name_or_path
        self.pretrained_generator_name_or_path = pretrained_generator_name_or_path
