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
""" TF 2.0 BORT model. """

from ...file_utils import add_start_docstrings
from ...utils import logging
from ..bert.modeling_tf_bert import (
    TFBertForMaskedLM,
    TFBertForMultipleChoice,
    TFBertForQuestionAnswering,
    TFBertForSequenceClassification,
    TFBertForTokenClassification,
    TFBertModel,
)
from .configuration_bort import BortConfig


logger = logging.get_logger(__name__)

_TOKENIZER_FOR_DOC = "RobertaTokenizer"

TF_BORT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # See all BORT models at https://huggingface.co/models?filter=bort
]

BORT_START_DOCSTRING = r"""

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
        config (:class:`~transformers.BortConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


@add_start_docstrings(
    "The bare BORT Model transformer outputting raw hidden-states without any specific head on top.",
    BORT_START_DOCSTRING,
)
class TFBortModel(TFBertModel):
    """
    This class overrides :class:`~transformers.TFBertModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples::
        >>> from transformers import TFBortModel, BortTokenizer

        >>> model = TFBortModel.from_pretrained("bort")
        >>> tokenizer = BortTokenizer.from_pretrained("bort")

        >>> inputs = tokenizer.encode_plus("The eastern endpoint of the canal is the Hubertusbrunnen.", return_tensors="tf")

        >>> outputs = model(**inputs)
        >>> hidden_states = outputs.last_hidden_state
    """

    config_class = BortConfig


@add_start_docstrings(
    """BORT Model with a `language modeling` head on top. """,
    BORT_START_DOCSTRING,
)
class TFBortForMaskedLM(TFBertForMaskedLM):
    """
    This class overrides :class:`~transformers.TFBertForMaskedLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples::
        >>> from transformers import TFBortForMaskedLM, BortTokenizer
        >>> import tensorflow as tf

        >>> model = TFBortForMaskedLM.from_pretrained("bort")
        >>> tokenizer = BortTokenizer.from_pretrained("bort")

        >>> inputs = tokenizer("The Nymphenburg Palace is <mask>.", return_tensors="tf")
        >>> labels = tokenizer("The Nymphenburg Palace is beautiful.", return_tensors="tf")["input_ids"]

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
    """

    config_class = BortConfig


@add_start_docstrings(
    """
    BORT Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BORT_START_DOCSTRING,
)
class TFBortForSequenceClassification(TFBertForSequenceClassification):
    """
    This class overrides :class:`~transformers.TFBertForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::
        >>> from transformers import TFBortForSequenceClassification, BortTokenizer
        >>> import tensorflow as tf

        >>> model = TFBortForSequenceClassification.from_pretrained("bort")
        >>> tokenizer = BortTokenizer.from_pretrained("bort")

        >>> inputs = tokenizer("The Nymphenburg Palace is beautiful.", return_tensors="tf")
        >>> inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1)) # Batch size 1

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
    """

    config_class = BortConfig


@add_start_docstrings(
    """
    BORT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BORT_START_DOCSTRING,
)
class TFBortForTokenClassification(TFBertForTokenClassification):
    """
    This class overrides :class:`~transformers.TFBertForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::
        >>> from transformers import TFBortForTokenClassification, BortTokenizer
        >>> import tensorflow as tf

        >>> model = TFBortForTokenClassification.from_pretrained("bort")
        >>> tokenizer = BortTokenizer.from_pretrained("bort")

        >>> inputs = tokenizer("The Nymphenburg Palace in Munich.", return_tensors="tf")
        >>> input_ids = inputs["input_ids"]
        >>> inputs["labels"] = tf.reshape(tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids))) # Batch size 1

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
    """

    config_class = BortConfig


@add_start_docstrings(
    """
    BORT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BORT_START_DOCSTRING,
)
class TFBortForMultipleChoice(TFBertForMultipleChoice):
    """
    This class overrides :class:`~transformers.TFBertForMultipleChoice`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::
        >>> from transformers import TFBortForMultipleChoice, BortTokenizer
        >>> import tensorflow as tf

        >>> model = TFBortForMultipleChoice.from_pretrained("bort")
        >>> tokenizer = BortTokenizer.from_pretrained("bort")

        >>> prompt = "The Nymphenburg Palace is a:"
        >>> choice0 = "Baroque palace."
        >>> choice1 = "Gothic palace."  # no!

        >>> encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='tf', padding=True)
        >>> inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
        >>> outputs = model(inputs)  # batch size is 1

        >>> # the linear classifier still needs to be trained
        >>> logits = outputs.logits
    """

    config_class = BortConfig


@add_start_docstrings(
    """
    BORT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BORT_START_DOCSTRING,
)
class TFBortForQuestionAnswering(TFBertForQuestionAnswering):
    """
    This class overrides :class:`~transformers.TFBertForQuestionAnswering`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::
        >>> from transformers import TFBortForQuestionAnswering, BortTokenizer
        >>> import tensorflow as tf

        >>> model = TFBortForQuestionAnswering.from_pretrained("bort")
        >>> tokenizer = BortTokenizer.from_pretrained("bort")

        >>> question, text = "Where is Nymphenburg Palace?", "Nymphenburg Palace is located in Munich"
        >>> input_dict = tokenizer(question, text, return_tensors='tf')
        >>> outputs = model(input_dict)
        >>> start_logits = outputs.start_logits
        >>> end_logits = outputs.end_logits

        >>> all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
        >>> answer = ' '.join(all_tokens[tf.math.argmax(start_logits, 1)[0] : tf.math.argmax(end_logits, 1)[0]+1])
    """

    config_class = BortConfig
