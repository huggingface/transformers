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


from .modeling_tf_bert import (
    TFBertModel,
    TFBertForMaskedLM,
    TFBertForSequenceClassification,
    TFBertForMultipleChoice,
    TFBertForTokenClassification,
    TFBertForQuestionAnswering,
)

from .file_utils import add_start_docstrings
from .configuration_herbert import HerbertConfig

logger = logging.getLogger(__name__)


HERBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
]


HERBERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.TFBertModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass.
    Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general
    usage and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having
        all the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors
        in the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Args:
        config (:class:`~transformers.HerbertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


@add_start_docstrings(
    "The bare HerBERT Model transformer outputting raw hidden-states without any specific head on top.",
    HERBERT_START_DOCSTRING,
)
class TFHerbertModel(TFBertModel):
    """
    This class overrides :class:`~transformers.TFBertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = HerbertConfig
    base_model_prefix = "herbert"


@add_start_docstrings(
    """HerBERT Model with a `language modeling` head on top. """, HERBERT_START_DOCSTRING,
)
class TFHerbertForMaskedLM(TFBertForMaskedLM):
    """
    This class overrides :class:`~transformers.TFBertForMaskedLM`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = HerbertConfig
    base_model_prefix = "herbert"


@add_start_docstrings(
    """HerBERT Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    HERBERT_START_DOCSTRING,
)
class TFHerbertForSequenceClassification(TFBertForSequenceClassification):
    """
    This class overrides :class:`~transformers.TFBertForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = HerbertConfig
    base_model_prefix = "herbert"


@add_start_docstrings(
    """HerBERT Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    HERBERT_START_DOCSTRING,
)
class TFHerbertForMultipleChoice(TFBertForMultipleChoice):
    """
    This class overrides :class:`~transformers.TFBertForMultipleChoice`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = HerbertConfig
    base_model_prefix = "herbert"


@add_start_docstrings(
    """HerBERT Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    HERBERT_START_DOCSTRING,
)
class TFHerbertForTokenClassification(TFBertForTokenClassification):
    """
    This class overrides :class:`~transformers.TFBertForTokenClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = HerbertConfig
    base_model_prefix = "herbert"


@add_start_docstrings(
    """HerBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits` """,
    HERBERT_START_DOCSTRING,
)
class TFHerbertForQuestionAnswering(TFBertForQuestionAnswering):
    """
    This class overrides :class:`~transformers.TFBertForQuestionAnswering`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = HerbertConfig
    base_model_prefix = "herbert"
