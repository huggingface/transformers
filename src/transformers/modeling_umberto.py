# coding=utf-8
# Copyright 2019 Inria, Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
# This code refers to the file modeling_camembert.py trying to replicate it
# not for copying it, but as reference

"""PyTorch UmBERTo model. """
# This code is referring to the Camembert code, just to simplify


import logging

from .configuration_umberto import UmbertoConfig
from .file_utils import add_start_docstrings

from .modeling_roberta import (
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)


logger = logging.getLogger(__name__)

UMBERTO_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "umberto-wikipedia-uncased-v1" : "https://mxmdownloads.s3.amazonaws.com/umberto/umberto-wikipedia-uncased-v1-pytorch_model.bin",
    "umberto-commoncrawl-cased-v1" : "https://mxmdownloads.s3.amazonaws.com/umberto/umberto-commoncrawl-cased-v1-pytorch_model.bin"
}


UMBERTO_START_DOCSTRING = r"""
"""

UMBERTO_INPUTS_DOCSTRING = r"""
    
"""


@add_start_docstrings(
    "",
    UMBERTO_START_DOCSTRING,
    UMBERTO_INPUTS_DOCSTRING,
)
class UmbertoModel(RobertaModel):
    r"""
    Examples::

        tokenizer = UmbertoTokenizer.from_pretrained('umberto-commoncrawl-cased-v1')
        model = UmbertoModel.from_pretrained('umberto-commoncrawl-cased-v1')
        input_ids = torch.tensor(tokenizer.encode("Umberto Eco è stato un grande scrittore")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    config_class = UmbertoConfig
    pretrained_model_archive_map = UMBERTO_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """umBERTo Model with a `language modeling` head on top. """,
    UMBERTO_START_DOCSTRING,
    UMBERTO_INPUTS_DOCSTRING,
)
class UmbertoForMaskedLM(RobertaForMaskedLM):
    r"""
        tokenizer = UmbertoTokenizer.from_pretrained('umberto-commoncrawl-cased-v1')
        model = UmbertoModel.from_pretrained('umberto-commoncrawl-cased-v1')
        input_ids = torch.tensor(tokenizer.encode("Umberto Eco è stato un grande scrittore")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    config_class = UmbertoConfig
    pretrained_model_archive_map = UMBERTO_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """umBERTo Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    UMBERTO_START_DOCSTRING,
    UMBERTO_INPUTS_DOCSTRING,
)
class UmbertoForSequenceClassification(RobertaForSequenceClassification):
    r"""
    Examples::

        tokenizer = UmbertoTokenizer.from_pretrained('umberto-commoncrawl-cased-v1')
        model = UmbertoModel.from_pretrained('umberto-commoncrawl-cased-v1')
        input_ids = torch.tensor(tokenizer.encode("Umberto Eco è stato un grande scrittore")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = UmbertoConfig
    pretrained_model_archive_map = UMBERTO_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """UmBERTo Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    UMBERTO_START_DOCSTRING,
    UMBERTO_INPUTS_DOCSTRING,
)
class UmbertoForMultipleChoice(RobertaForMultipleChoice):
    r"""
    
    Examples::

        tokenizer = UmbertoTokenizer.from_pretrained('umberto-commoncrawl-cased-v1')
        model = UmbertoModel.from_pretrained('umberto-commoncrawl-cased-v1')
        choices = ["Umberto Eco è stato un grande scrittore", "Umberto Eco è stato un grande autore"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    config_class = UmbertoConfig
    pretrained_model_archive_map = UMBERTO_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """umBERTo Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    UMBERTO_START_DOCSTRING,
    UMBERTO_INPUTS_DOCSTRING,
)
class UmbertoForTokenClassification(RobertaForTokenClassification):
    r"""
    
    Examples::

        tokenizer = UmbertoTokenizer.from_pretrained('umberto-commoncrawl-cased-v1')
        model = UmbertoModel.from_pretrained('umberto-commoncrawl-cased-v1')
        input_ids = torch.tensor(tokenizer.encode("Umberto Eco è stato un grande scrittore", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    config_class = UmbertoConfig
    pretrained_model_archive_map = UMBERTO_PRETRAINED_MODEL_ARCHIVE_MAP
