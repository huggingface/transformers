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
from typing import Iterable, List, Optional, Tuple

import numpy as np

from ...file_utils import cached_path, is_datasets_available, is_faiss_available, is_remote_url, requires_backends
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import logging
from .configuration_rag import RagConfig
from .tokenization_rag import RagTokenizer


if is_datasets_available():
    from datasets import Dataset, load_dataset, load_from_disk

if is_faiss_available():
    import faiss


logger = logging.get_logger(__name__)


LEGACY_INDEX_PATH = "https://storage.googleapis.com/huggingface-nlp/datasets/wiki_dpr/"



class RealmRetriever:

    def __init__(self, config, tokenizer, index=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.index

    @classmethod
    def from_pretrained(cls, retriever_name_or_path, indexed_dataset=None, **kwargs):
        config = kwargs.pop("config", None) or RealmConfig.from_pretrained(retriever_name_or_path, **kwargs)
        tokenizer = RealmTokenizer.from_pretrained(retriever_name_or_path, config=config)

        # logic to load tf.records (should probs put it in `datasets`)
        index = None

        return cls(
            config,
            tokenizer=tokenizer,
            index=index,
        )

    def save_pretrained(self, save_directory):
        # save index here

        self.config.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def __call__(
        self,
        question_input_ids: List[List[int]],
        question_hidden_states: np.ndarray,
        prefix=None,
        n_docs=None,
        return_tensors=None,
    ) -> BatchEncoding:

        n_docs = n_docs if n_docs is not None else self.n_docs
        prefix = prefix if prefix is not None else self.config.generator.prefix
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)

        input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
        context_input_ids, context_attention_mask = self.postprocess_docs(
            docs, input_strings, prefix, n_docs, return_tensors=return_tensors
        )

        if self.return_tokenized_docs:
            retrived_doc_text = []
            retrived_doc_title = []

            for b_idx in range(len(docs)):
                for doc_idx in range(n_docs):
                    retrived_doc_text.append(docs[b_idx]["text"][doc_idx])
                    retrived_doc_title.append(docs[b_idx]["title"][doc_idx])

            tokenized_docs = self.ctx_encoder_tokenizer(
                retrived_doc_title,
                retrived_doc_text,
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
