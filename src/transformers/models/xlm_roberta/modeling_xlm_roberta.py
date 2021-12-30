# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
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
"""PyTorch XLM-RoBERTa model."""

from ...file_utils import add_start_docstrings
from ...utils import logging
from ..roberta.modeling_roberta import (
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)
from .configuration_xlm_roberta import XLMRobertaConfig


logger = logging.get_logger(__name__)

XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "xlm-roberta-large-finetuned-conll02-dutch",
    "xlm-roberta-large-finetuned-conll02-spanish",
    "xlm-roberta-large-finetuned-conll03-english",
    "xlm-roberta-large-finetuned-conll03-german",
    # See all XLM-RoBERTa models at https://huggingface.co/models?filter=xlm-roberta
]


XLM_ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`XLMRobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaModel(RobertaModel):
    """
    This class overrides [`RobertaModel`]. Please check the superclass for the appropriate documentation alongside
    usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    "XLM-RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForCausalLM(RobertaForCausalLM):
    """
    This class overrides [`RobertaForCausalLM`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """XLM-RoBERTa Model with a `language modeling` head on top.""",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForMaskedLM(RobertaForMaskedLM):
    """
    This class overrides [`RobertaForMaskedLM`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    """
    This class overrides [`RobertaForSequenceClassification`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForMultipleChoice(RobertaForMultipleChoice):
    """
    This class overrides [`RobertaForMultipleChoice`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForTokenClassification(RobertaForTokenClassification):
    """
    This class overrides [`RobertaForTokenClassification`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    """
    This class overrides [`RobertaForQuestionAnswering`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = XLMRobertaConfig
