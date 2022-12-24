# coding=utf-8
# Copyright 2022, The ATLAS Authors and The HuggingFace Inc. team.
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
"""ATLAS model implementation."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from functools import reduce


from ...configuration_utils import PretrainedConfig
from ...generation import BeamSearchScorer, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_atlas import AtlasConfig
from .tokenization_atlas import AtlasTokenizer

from .retriever import Contriever, UntiedDualEncoderRetriever, DualEncoderRetriever
from .fid import FiD


from ...utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends

if is_datasets_available():
    from datasets import Dataset, load_dataset, load_from_disk

if is_faiss_available():
    import faiss

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "AtlasConfig"

class AtlasPreTrainedModel(PreTrainedModel):
    config_class = AtlasConfig
    base_model_prefix = "atlas"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)
    
    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str = None,
        generator_pretrained_model_name_or_path: str = None,
        index: Dataset = None,
        **kwargs
    ) -> PreTrainedModel:
        pass
        

class AtlasModel(AtlasPreTrainedModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        query_passage_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        index: Optional[Dataset] = None,
        **kwargs,
    ):
        requires_backends(self, ["datasets", "faiss"])
        assert config is not None or (
            query_passage_encoder is not None and generator is not None
        ), "Either a configuration or an query_passage_encoder and a generator has to be provided."

        if config is None:
            config = AtlasConfig.from_pretrained_query_passage_generator(
                query_passage_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        
        super().__init__(config)
        
        if query_passage_encoder is None:
            from ..auto.modeling_auto import AutoModel
            contriever = Contriever(config.query_passage_encoder)
            if config.query_side_retriever_training:
                query_passage_encoder = UntiedDualEncoderRetriever(config, contriever)
            else:
                query_passage_encoder = DualEncoderRetriever(config, contriever)

        if generator is None:
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM
            generator = FiD(config.generator)

        self.index = index
        if self.index is not None:
            self.set_index(index)

        self.query_passage_encoder = query_passage_encoder
        self.generator = generator

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

            hidden_states = self.query_passage_encoder.embed_passages(
                input_ids=tokenized["input_ids"].to(self.device),
                attention_mask=tokenized["attention_mask"].to(self.device)
            )
            examples['embeddings'] = hidden_states.cpu().detach().numpy()
            return examples

        self.index = self.index.index.map(reindex, batched=True, batch_size=batch_size)
        self.index.add_faiss_index("embeddings", device=device, string_factory=string_factory, metric_type=metric_type)

    def forward(
        self,
        queries,
        target,
        target_tokens,
        topk,
    ):
        bsz = len(queries)

        queries_tokens = self.query_encoder_tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512)

        query_hidden_states = self.query_passage_encoder(input_ids=queries_tokens["input_ids"].to(self.device), attention_mask=queries_tokens["attention_mask"].to(self.device))

        query_hidden_states = query_hidden_states.cpu().detach().numpy()
        _, passage_ids = self.index.search_batch("embeddings", query_hidden_states, topk)
        docs = [self.index[[i for i in indices if i >= 0]] for indices in passage_ids]
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

        reader_tokens = encode_passages(passages, self.generator_tokenizer, 512)
        labels = self.generator_tokenizer(target, return_tensors="pt", padding=True, truncation=True, max_length=512)['input_ids'].to(self.device)
        labels[labels == self.generator_tokenizer.pad_token_id] = -100

        reader_ids = reader_tokens["input_ids"].to(self.device)  # FIXME (Not added by ae99, TODO figure out why)
        reader_mask = reader_tokens["attention_mask"].bool().to(self.device)

        n_context_training = min(topk, reader_ids.size(1))
        cfg = self.generator.encoder.config
        cfg.bsz = reader_ids.size(0)
        cfg.n_context = n_context_training

        reader_ids_training = reader_ids[:, :n_context_training].contiguous()
        reader_mask_training = reader_mask[:, :n_context_training].contiguous()

        reader_ids_training = reader_ids_training.view(reader_ids.size(0), -1)
        reader_mask_training = reader_mask_training.view(reader_mask.size(0), -1)

        return self.generator(
            input_ids=reader_ids_training,
            attention_mask=reader_mask_training,
            decoder_input_ids=None,
            labels=labels,
            use_cache=False,
        )

