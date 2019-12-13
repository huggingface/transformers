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
""" PyTorch - TF 2.0 general utilities."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import os
import re
import numpy

logger = logging.getLogger(__name__)

def convert_tf_weight_name_to_pt_weight_name(tf_name, start_prefix_to_remove=''):
    """ Convert a TF 2.0 model variable name in a pytorch model weight name.

        Conventions for TF2.0 scopes -> PyTorch attribute names conversions:
            - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
            - '_._' is replaced by a new level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)

        return tuple with:
            - pytorch model weight name
            - transpose: boolean indicating weither TF2.0 and PyTorch weights matrices are transposed with regards to each other
    """
    tf_name = tf_name.replace(':0', '')                       # device ids
    tf_name = re.sub(r'/[^/]*___([^/]*)/', r'/\1/', tf_name)  # '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
    tf_name = tf_name.replace('_._', '/')                     # '_._' is replaced by a level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)
    tf_name = re.sub(r'//+', '/', tf_name)                    # Remove empty levels at the end
    tf_name = tf_name.split('/')                              # Convert from TF2.0 '/' separators to PyTorch '.' separators
    tf_name = tf_name[1:]                                     # Remove level zero

    # When should we transpose the weights
    transpose = bool(tf_name[-1] == 'kernel' or 'emb_projs' in tf_name or 'out_projs' in tf_name)

    # Convert standard TF2.0 names in PyTorch names
    if tf_name[-1] == 'kernel' or tf_name[-1] == 'embeddings' or tf_name[-1] == 'gamma':
        tf_name[-1] = 'weight'
    if tf_name[-1] == 'beta':
        tf_name[-1] = 'bias'

    # Remove prefix if needed
    tf_name = '.'.join(tf_name)
    if start_prefix_to_remove:
        tf_name = tf_name.replace(start_prefix_to_remove, '', 1)

    return tf_name, transpose


#####################
### PyTorch => TF 2.0

def load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path, tf_inputs=None, allow_missing_keys=False):
    """ Load pytorch checkpoints in a TF 2.0 model
    """
    try:
        import tensorflow as tf
        import torch
    except ImportError as e:
        logger.error("Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.")
        raise e

    pt_path = os.path.abspath(pytorch_checkpoint_path)
    logger.info("Loading PyTorch weights from {}".format(pt_path))

    pt_state_dict = torch.load(pt_path, map_location='cpu')

    return load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys)


def load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=None, allow_missing_keys=False):
    """ Load pytorch checkpoints in a TF 2.0 model
    """
    pt_state_dict = pt_model.state_dict()

    return load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys)


def load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=None, allow_missing_keys=False):
    """ Load pytorch state_dict in a TF 2.0 model.
    """
    try:
        import torch
        import tensorflow as tf
        from tensorflow.python.keras import backend as K
    except ImportError as e:
        logger.error("Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.")
        raise e

    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs

    if tf_inputs is not None:
        tfo = tf_model(tf_inputs, training=False)  # Make sure model is built

    # Adapt state dict - TODO remove this and update the AWS weights files instead
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in pt_state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        # DialoGPT format
        if key == 'lm_head.decoder.weight':
            new_key = 'lm_head.weight'
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        pt_state_dict[new_key] = pt_state_dict.pop(old_key)

    # Make sure we are able to load PyTorch base models as well as derived models (with heads)
    # TF models always have a prefix, some of PyTorch models (base ones) don't
    start_prefix_to_remove = ''
    if not any(s.startswith(tf_model.base_model_prefix) for s in pt_state_dict.keys()):
        start_prefix_to_remove = tf_model.base_model_prefix + '.'

    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights

    weight_value_tuples = []
    all_pytorch_weights = set(list(pt_state_dict.keys()))
    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.name
        name, transpose = convert_tf_weight_name_to_pt_weight_name(sw_name, start_prefix_to_remove=start_prefix_to_remove)

        # Find associated numpy array in pytorch model state dict
        assert name in pt_state_dict, "{} not found in PyTorch model".format(name)
        array = pt_state_dict[name].numpy()

        if transpose:
            array = numpy.transpose(array)

        if len(symbolic_weight.shape) < len(array.shape):
            array = numpy.squeeze(array)
        elif len(symbolic_weight.shape) > len(array.shape):
            array = numpy.expand_dims(array, axis=0)

        try:
            assert list(symbolic_weight.shape) == list(array.shape)
        except AssertionError as e:
            e.args += (symbolic_weight.shape, array.shape)
            raise e

        logger.info("Initialize TF weight {}".format(symbolic_weight.name))

        weight_value_tuples.append((symbolic_weight, array))
        all_pytorch_weights.discard(name)

    K.batch_set_value(weight_value_tuples)

    if tf_inputs is not None:
        tfo = tf_model(tf_inputs, training=False)  # Make sure restore ops are run

    logger.info("Weights or buffers not loaded from PyTorch model: {}".format(all_pytorch_weights))

    return tf_model


#####################
### TF 2.0 => PyTorch

def load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path, tf_inputs=None, allow_missing_keys=False):
    """ Load TF 2.0 HDF5 checkpoint in a PyTorch model
        We use HDF5 to easily do transfer learning
        (see https://github.com/tensorflow/tensorflow/blob/ee16fcac960ae660e0e4496658a366e2f745e1f0/tensorflow/python/keras/engine/network.py#L1352-L1357).
    """
    try:
        import tensorflow as tf
        import torch
    except ImportError as e:
        logger.error("Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.")
        raise e

    import transformers

    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Loading TensorFlow weights from {}".format(tf_checkpoint_path))

    # Instantiate and load the associated TF 2.0 model
    tf_model_class_name = "TF" + pt_model.__class__.__name__  # Add "TF" at the beggining
    tf_model_class = getattr(transformers, tf_model_class_name)
    tf_model = tf_model_class(pt_model.config)

    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs

    if tf_inputs is not None:
        tfo = tf_model(tf_inputs, training=False)  # Make sure model is built

    tf_model.load_weights(tf_checkpoint_path, by_name=True)

    return load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=allow_missing_keys)

def load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=False):
    """ Load TF 2.0 model in a pytorch model
    """
    weights = tf_model.weights

    return load_tf2_weights_in_pytorch_model(pt_model, weights, allow_missing_keys=allow_missing_keys)


def load_tf2_weights_in_pytorch_model(pt_model, tf_weights, allow_missing_keys=False):
    """ Load TF2.0 symbolic weights in a PyTorch model
    """
    try:
        import tensorflow as tf
        import torch
    except ImportError as e:
        logger.error("Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.")
        raise e

    new_pt_params_dict = {}
    current_pt_params_dict = dict(pt_model.named_parameters())

    # Make sure we are able to load PyTorch base models as well as derived models (with heads)
    # TF models always have a prefix, some of PyTorch models (base ones) don't
    start_prefix_to_remove = ''
    if not any(s.startswith(pt_model.base_model_prefix) for s in current_pt_params_dict.keys()):
        start_prefix_to_remove = pt_model.base_model_prefix + '.'

    # Build a map from potential PyTorch weight names to TF 2.0 Variables
    tf_weights_map = {}
    for tf_weight in tf_weights:
        pt_name, transpose = convert_tf_weight_name_to_pt_weight_name(tf_weight.name, start_prefix_to_remove=start_prefix_to_remove)
        tf_weights_map[pt_name] = (tf_weight.numpy(), transpose)

    all_tf_weights = set(list(tf_weights_map.keys()))
    loaded_pt_weights_data_ptr = {}
    for pt_weight_name, pt_weight in current_pt_params_dict.items():
        # Handle PyTorch shared weight ()not duplicated in TF 2.0
        if pt_weight.data_ptr() in loaded_pt_weights_data_ptr:
            new_pt_params_dict[pt_weight_name] = loaded_pt_weights_data_ptr[pt_weight.data_ptr()]
            continue

        # Find associated numpy array in pytorch model state dict
        if pt_weight_name not in tf_weights_map:
            raise ValueError("{} not found in TF 2.0 model".format(pt_weight_name))

        array, transpose = tf_weights_map[pt_weight_name]

        if transpose:
            array = numpy.transpose(array)

        if len(pt_weight.shape) < len(array.shape):
            array = numpy.squeeze(array)
        elif len(pt_weight.shape) > len(array.shape):
            array = numpy.expand_dims(array, axis=0)

        try:
            assert list(pt_weight.shape) == list(array.shape)
        except AssertionError as e:
            e.args += (pt_weight.shape, array.shape)
            raise e

        logger.info("Initialize PyTorch weight {}".format(pt_weight_name))

        new_pt_params_dict[pt_weight_name] = torch.from_numpy(array)
        loaded_pt_weights_data_ptr[pt_weight.data_ptr()] = torch.from_numpy(array)
        all_tf_weights.discard(pt_weight_name)

    missing_keys, unexpected_keys = pt_model.load_state_dict(new_pt_params_dict, strict=False)

    if len(missing_keys) > 0:
        logger.info("Weights of {} not initialized from TF 2.0 model: {}".format(
            pt_model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info("Weights from TF 2.0 model not used in {}: {}".format(
            pt_model.__class__.__name__, unexpected_keys))

    logger.info("Weights or buffers not loaded from TF 2.0 model: {}".format(all_tf_weights))

    return pt_model
