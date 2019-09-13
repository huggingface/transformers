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

logger = logging.getLogger(__name__)


def load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path, tf_inputs=None):
    """ Load pytorch checkpoints in a TF 2.0 model
        Conventions for TF2.0 scopes -> PyTorch attribute names conversions:
            - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
            - '_._' is replaced by a new level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)
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

    return load_pytorch_state_dict_in_tf2_model(tf_model, pt_state_dict, tf_inputs=tf_inputs)


def load_pytorch_state_dict_in_tf2_model(tf_model, pt_state_dict, tf_inputs=None):
    """ Load pytorch state_dict in a TF 2.0 model.
        Conventions for TF2.0 scopes -> PyTorch attribute names conversions:
            - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
            - '_._' is replaced by a level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)
    """
    try:
        import re
        import torch
        import numpy
        from tensorflow.python.keras import backend as K
    except ImportError as e:
        logger.error("Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.")
        raise e

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
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        pt_state_dict[new_key] = pt_state_dict.pop(old_key)

    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights

    weight_value_tuples = []
    all_pytorch_weights = set(list(pt_state_dict.keys()))
    for symbolic_weight in symbolic_weights:
        name = symbolic_weight.name
        name = name.replace(':0', '')                       # device ids
        name = re.sub(r'/[^/]*___([^/]*)/', r'/\1/', name)  # '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
        name = name.replace('_._', '/')                     # '_._' is replaced by a level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)
        name = re.sub(r'//+', '/', name)                    # Remove empty levels at the end
        name = name.split('/')                              # Convert from TF2.0 '/' separators to PyTorch '.' separators
        name = name[1:]                                     # Remove level zero

        # When should we transpose the weights
        transpose = bool(name[-1] == 'kernel' or 'emb_projs' in name or 'out_projs' in name)

        # Convert standard TF2.0 names in PyTorch names
        if name[-1] == 'kernel' or name[-1] == 'embeddings' or name[-1] == 'gamma':
            name[-1] = 'weight'
        if name[-1] == 'beta':
            name[-1] = 'bias'

        name = '.'.join(name)
        assert name in pt_state_dict, "{} not found in PyTorch model".format(name)
        array = pt_state_dict[name].numpy()

        if transpose:
            array = numpy.transpose(array)

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


def load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path):
    """ Load TF 2.0 HDF5 checkpoint in a PyTorch model
        We use HDF5 to easily do transfer learning
        (see https://github.com/tensorflow/tensorflow/blob/ee16fcac960ae660e0e4496658a366e2f745e1f0/tensorflow/python/keras/engine/network.py#L1352-L1357).
    """
    raise NotImplementedError

def load_tf2_weights_in_pytorch_model(pt_model, tf_model):
    """ Load TF2.0 symbolic weights in a PyTorch model
    """
    raise NotImplementedError
