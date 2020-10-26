# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" TF 2.0 CamemBERT model. """

from .configuration_camembert import CamembertConfig
from .file_utils import add_start_docstrings
from .modeling_tf_roberta import (
    TFRobertaForMaskedLM,
    TFRobertaForMultipleChoice,
    TFRobertaForQuestionAnswering,
    TFRobertaForSequenceClassification,
    TFRobertaForTokenClassification,
    TFRobertaModel,
)
from .utils import logging


logger = logging.get_logger(__name__)

TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # See all CamemBERT models at https://huggingface.co/models?filter=camembert
]


CAMEMBERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass. Use
    it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having all
        the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in
        the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Parameters:
        config (:class:`~transformers.CamembertConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


@add_start_docstrings(
    "The bare CamemBERT Model transformer outputting raw hidden-states without any specific head on top.",
    CAMEMBERT_START_DOCSTRING,
)
class TFCamembertModel(TFRobertaModel):
    """
    This class overrides :class:`~transformers.TFRobertaModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = CamembertConfig


@add_start_docstrings(
    """CamemBERT Model with a `language modeling` head on top. """,
    CAMEMBERT_START_DOCSTRING,
)
class TFCamembertForMaskedLM(TFRobertaForMaskedLM):
    """
    This class overrides :class:`~transformers.TFRobertaForMaskedLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = CamembertConfig


@add_start_docstrings(
    """
    CamemBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
class TFCamembertForSequenceClassification(TFRobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.TFRobertaForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = CamembertConfig


@add_start_docstrings(
    """
    CamemBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
class TFCamembertForTokenClassification(TFRobertaForTokenClassification):
    """
    This class overrides :class:`~transformers.TFRobertaForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = CamembertConfig


@add_start_docstrings(
    """
    CamemBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
class TFCamembertForMultipleChoice(TFRobertaForMultipleChoice):
    """
    This class overrides :class:`~transformers.TFRobertaForMultipleChoice`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = CamembertConfig


@add_start_docstrings(
    """
    CamemBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    CAMEMBERT_START_DOCSTRING,
)
class TFCamembertForQuestionAnswering(TFRobertaForQuestionAnswering):
    """
    This class overrides :class:`~transformers.TFRobertaForQuestionAnswering`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = CamembertConfig
