# coding=utf-8
# Copyright 2020, The HuggingFace Inc. team.
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
"""PyTorch BORT model. """

from ...file_utils import add_start_docstrings
from ...utils import logging
from ..bert.modeling_bert import (
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
)
from .configuration_bort import BortConfig


logger = logging.get_logger(__name__)

_TOKENIZER_FOR_DOC = "RobertaTokenizer"

BORT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # See all BORT models at https://huggingface.co/models?filter=bort
]

BORT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BortConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


@add_start_docstrings(
    "The bare BORT Model transformer outputting raw hidden-states without any specific head on top.",
    BORT_START_DOCSTRING,
)
class BortModel(BertModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = BortConfig


@add_start_docstrings(
    """BORT Model with a `language modeling` head on top. """,
    BORT_START_DOCSTRING,
)
class BortForMaskedLM(BertForMaskedLM):
    """
    This class overrides :class:`~transformers.BertForMaskedLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = BortConfig


@add_start_docstrings(
    """
    BORT Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BORT_START_DOCSTRING,
)
class BortForSequenceClassification(BertForSequenceClassification):
    """
    This class overrides :class:`~transformers.BertForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = BortConfig


@add_start_docstrings(
    """
    BORT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BORT_START_DOCSTRING,
)
class BortForMultipleChoice(BertForMultipleChoice):
    """
    This class overrides :class:`~transformers.BertForMultipleChoice`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = BortConfig


@add_start_docstrings(
    """
    BORT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BORT_START_DOCSTRING,
)
class BortForTokenClassification(BertForTokenClassification):
    """
    This class overrides :class:`~transformers.BertForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = BortConfig


@add_start_docstrings(
    """
    BORT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`
    """,
    BORT_START_DOCSTRING,
)
class BortForQuestionAnswering(BertForQuestionAnswering):
    """
    This class overrides :class:`~transformers.BertForQuestionAnswering`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = BortConfig
