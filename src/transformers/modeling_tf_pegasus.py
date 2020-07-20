""" TF 2.0 PEGASUS model. """


import copy
import itertools
import logging
import math

import tensorflow as tf

from .configuration_pegasus import PegasusConfig
from .file_utils import DUMMY_INPUTS, DUMMY_MASK, add_start_docstrings, add_start_docstrings_to_callable
from .modeling_tf_utils import (
    TFPreTrainedModel,
    TFSharedEmbeddings,
    cast_bool_to_primitive,
    keras_serializable,
    shape_list,
)
from .tokenization_utils import BatchEncoding


logger = logging.getLogger(__name__)

TF_PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "pegasus-large",
    # See all PEGASUS models at https://huggingface.co/models?filter=pegasus
]

####################################################
# TF 2.0 Models are constructed using Keras imperative API by sub-classing
# - tf.keras.layers.Layer for the layers and
# - TFPreTrainedModel for the models (it-self a sub-class of tf.keras.Model)
####################################################


####################################################
# TFPegasusPreTrainedModel is a sub-class of tf.keras.Model
# which take care of loading and saving pretrained weights
# and various common utilities.
# Here you just need to specify a few (self-explanatory)
# pointers for your model.
####################################################
class TFPegasusPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = PegasusConfig
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        inputs = tf.constant(DUMMY_INPUTS)
        dummy_inputs = {
            "inputs": inputs,
        }
        return dummy_inputs

#TODO: add docstring
PEGASUS_START_DOCSTRING = r"""PEGASUS start"""

PEGASUS_INPUTS_DOCSTRING = r"""PEGASUS inputs"""


@add_start_docstrings(
    "The PEGASUS Model transformer",
    PEGASUS_START_DOCSTRING,
)
class TFPegasusModel(TFPegasusPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        print("init")

    @add_start_docstrings_to_callable(PEGASUS_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        print("call")
        return tf.zeros((2, 2)), tf.zeros((2, 2)), tf.zeros((2, 2))
