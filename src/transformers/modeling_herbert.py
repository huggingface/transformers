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
"""PyTorch HerBERT model. """

import logging


from .modeling_bert import (
    BertModel,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForMultipleChoice,
    BertForTokenClassification,
    BertForQuestionAnswering,
)

from .file_utils import add_start_docstrings
from transformers.configuration_herbert import HerbertConfig

logger = logging.getLogger(__name__)


HERBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "allegro/herbert-base-cased",
    "allegro/herbert-large-cased",
]


HERBERT_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.HerbertConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


@add_start_docstrings(
    "The bare HerBERT Model transformer outputting raw hidden-states without any specific head on top.",
    HERBERT_START_DOCSTRING,
)
class HerbertModel(BertModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = HerbertConfig


@add_start_docstrings(
    """HerBERT Model with a `language modeling` head on top. """, HERBERT_START_DOCSTRING,
)
class HerbertForMaskedLM(BertForMaskedLM):
    """
    This class overrides :class:`~transformers.BertForMaskedLM`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = HerbertConfig


@add_start_docstrings(
    """HerBERT Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    HERBERT_START_DOCSTRING,
)
class HerbertForSequenceClassification(BertForSequenceClassification):
    """
    This class overrides :class:`~transformers.BertForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = HerbertConfig


@add_start_docstrings(
    """HerBERT Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    HERBERT_START_DOCSTRING,
)
class HerbertForMultipleChoice(BertForMultipleChoice):
    """
    This class overrides :class:`~transformers.BertForMultipleChoice`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = HerbertConfig


@add_start_docstrings(
    """HerBERT Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    HERBERT_START_DOCSTRING,
)
class HerbertForTokenClassification(BertForTokenClassification):
    """
    This class overrides :class:`~transformers.BertForTokenClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = HerbertConfig


@add_start_docstrings(
    """HerBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits` """,
    HERBERT_START_DOCSTRING,
)
class HerbertForQuestionAnswering(BertForQuestionAnswering):
    """
    This class overrides :class:`~transformers.BertForQuestionAnswering`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = HerbertConfig
