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
"""PyTorch CamemBERT model."""

from ...utils import add_start_docstrings, logging
from ..roberta.modeling_roberta import (
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)
from .configuration_camembert import CamembertConfig


logger = logging.get_logger(__name__)

_TOKENIZER_FOR_DOC = "CamembertTokenizer"

CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "camembert-base",
    "Musixmatch/umberto-commoncrawl-cased-v1",
    "Musixmatch/umberto-wikipedia-uncased-v1",
    # See all CamemBERT models at https://huggingface.co/models?filter=camembert
]

CAMEMBERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CamembertConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare CamemBERT Model transformer outputting raw hidden-states without any specific head on top.",
    CAMEMBERT_START_DOCSTRING,
)
class CamembertModel(RobertaModel):
    """
    This class overrides [`RobertaModel`]. Please check the superclass for the appropriate documentation alongside
    usage examples.
    """

    config_class = CamembertConfig


@add_start_docstrings(
    """CamemBERT Model with a `language modeling` head on top.""",
    CAMEMBERT_START_DOCSTRING,
)
class CamembertForMaskedLM(RobertaForMaskedLM):
    """
    This class overrides [`RobertaForMaskedLM`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = CamembertConfig


@add_start_docstrings(
    """
    CamemBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
class CamembertForSequenceClassification(RobertaForSequenceClassification):
    """
    This class overrides [`RobertaForSequenceClassification`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = CamembertConfig


@add_start_docstrings(
    """
    CamemBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
class CamembertForMultipleChoice(RobertaForMultipleChoice):
    """
    This class overrides [`RobertaForMultipleChoice`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = CamembertConfig


@add_start_docstrings(
    """
    CamemBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
class CamembertForTokenClassification(RobertaForTokenClassification):
    """
    This class overrides [`RobertaForTokenClassification`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = CamembertConfig


@add_start_docstrings(
    """
    CamemBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`
    """,
    CAMEMBERT_START_DOCSTRING,
)
class CamembertForQuestionAnswering(RobertaForQuestionAnswering):
    """
    This class overrides [`RobertaForQuestionAnswering`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = CamembertConfig


@add_start_docstrings(
    """CamemBERT Model with a `language modeling` head on top for CLM fine-tuning.""", CAMEMBERT_START_DOCSTRING
)
class CamembertForCausalLM(RobertaForCausalLM):
    """
    This class overrides [`RobertaForCausalLM`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = CamembertConfig
