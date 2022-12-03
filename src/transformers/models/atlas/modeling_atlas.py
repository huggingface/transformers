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


from ...configuration_utils import PretrainedConfig
from ...generation import BeamSearchScorer, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_atlas import AtlasConfig
from .tokenization_atlas import AtlasTokenizer

from ...utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends

if is_datasets_available():
    from datasets import Dataset, load_dataset, load_from_disk

if is_faiss_available():
    import faiss

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "AtlasConfig"

class AtlasPreTrainedModel(PreTrainedModel):
    config_class = AtlasConfig
    

class AtlasModel(AtlasPreTrainedModel):
    def __init__(self, config, query_passage_encoder, generator, dataset: Dataset, tokenizer: AtlasTokenizer):
        super().__init__(config)
        requires_backends(self, ["datasets", "faiss"])

        self.index = dataset
        self.config = config
        self.query_passage_encoder = query_passage_encoder # UntiedDualEncoder
        self.generator = generator # FiD
        self.generator_tokenizer = tokenizer.generator
        self.query_encoder_tokenizer = tokenizer.query_encoder
        

    def forward(
        self,
        querys,
        target,
        target_tokens,
        topk,
    ):
        bsz = len(querys)
        

        querys_tokens = self.query_encoder_tokenizer(querys, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        query_hidden_states = self.query_passage_encoder(input_ids=querys_tokens["input_ids"], attention_mask=querys_tokens["attention_mask"])
        print(query_hidden_states.shape)

        query_hidden_states = query_hidden_states.cpu().detach().numpy()

        _, passage_ids = self.index.search_batch("embeddings", query_hidden_states, topk)

        
        docs = [self.index[[i for i in indices if i >= 0]] for indices in passage_ids]

        print(docs)

