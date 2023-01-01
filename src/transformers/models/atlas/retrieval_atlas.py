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
        # todo: for pre-training, we need to skip retrieval of the passage currently in the generator so it can't cheat
        ignore_index: Optional[int] = None,
    ):
        _, passage_ids = self.index.search_batch("embeddings", retriever_hidden_states, topk)

        docs = [self.index[[i for i in indices if i >= 0]] for indices in passage_ids]

        passages = self._format_docs(docs, generator_input_ids)

        generator_tokens = self._encode_passages(passages, 512)
        return generator_tokens

    def _format_docs(self, docs: List[str], generator_input_ids: List[int]):
        # todo: possible to re-use tokenized generator input ids here if non-complex formatting is used
        queries = self.generator_tokenizer.batch_decode(generator_input_ids, skip_special_tokens=True)
        # todo: this should be configurable with a string like "{query} context: {passage}" as input
        passages = [[f'{queries[i]} context: {passage}' for passage in doc["text"]] for i, doc in enumerate(docs)]

        return passages

    def _encode_passages(self, batch, max_length):
        # todo: possible to pre-tokenize passages and simply add padding on retrieval
        bsz = len(batch)
        n = max([len(example) for example in batch])
        batch = [example + [""] *  (n - len(example)) for example in batch]
        batch = reduce(lambda a, b: a + b, batch)
        tokens = self.generator_tokenizer(
            batch,
            # Max length padding is needed to reproduce original implementation, but has a performance cost
            # padding="max_length",
            padding=True,
            max_length=max_length,
            return_tensors="pt",
            truncation=True,
        )
        tokens = {k: v.view(bsz, n, -1) for k, v in tokens.items()}
        
        return tokens
