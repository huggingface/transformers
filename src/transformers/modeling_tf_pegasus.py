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
    "t5-large",
    # See all T5 models at https://huggingface.co/models?filter=pegasus
]

####################################################
# TF 2.0 Models are constructed using Keras imperative API by sub-classing
# - tf.keras.layers.Layer for the layers and
# - TFPreTrainedModel for the models (it-self a sub-class of tf.keras.Model)
####################################################

