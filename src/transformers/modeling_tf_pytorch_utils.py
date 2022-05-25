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


import os
import re

import numpy

from .utils import ExplicitEnum, logging


logger = logging.get_logger(__name__)


class TransposeType(ExplicitEnum):
    """
    Possible ...
    """

    NO = "no"
    SIMPLE = "simple"
    CONV1D = "conv1d"
    CONV2D = "conv2d"


def convert_tf_weight_name_to_pt_weight_name(tf_name, start_prefix_to_remove="", tf_weight_shape=None):
    """
    Convert a TF 2.0 model variable name in a pytorch model weight name.

    Conventions for TF2.0 scopes -> PyTorch attribute names conversions:

        - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
        - '_._' is replaced by a new level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)

    return tuple with:

        - pytorch model weight name
        - transpose: `TransposeType` member indicating whether and how TF2.0 and PyTorch weights matrices should be
          transposed with regards to each other
    """
    tf_name = tf_name.replace(":0", "")  # device ids
    tf_name = re.sub(
        r"/[^/]*___([^/]*)/", r"/\1/", tf_name
    )  # '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
    tf_name = tf_name.replace(
        "_._", "/"
    )  # '_._' is replaced by a level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)
    tf_name = re.sub(r"//+", "/", tf_name)  # Remove empty levels at the end
    tf_name = tf_name.split("/")  # Convert from TF2.0 '/' separators to PyTorch '.' separators
    # Some weights have a single name without "/" such as final_logits_bias in BART
    if len(tf_name) > 1:
        tf_name = tf_name[1:]  # Remove level zero

    # When should we transpose the weights
    if tf_name[-1] == "kernel" and tf_weight_shape is not None and tf_weight_shape.rank == 4:
        transpose = TransposeType.CONV2D
    elif tf_name[-1] == "kernel" and tf_weight_shape is not None and tf_weight_shape.rank == 3:
        transpose = TransposeType.CONV1D
    elif bool(
        tf_name[-1] in ["kernel", "pointwise_kernel", "depthwise_kernel"]
        or "emb_projs" in tf_name
        or "out_projs" in tf_name
    ):
        transpose = TransposeType.SIMPLE
    else:
        transpose = TransposeType.NO

    # Convert standard TF2.0 names in PyTorch names
    if tf_name[-1] == "kernel" or tf_name[-1] == "embeddings" or tf_name[-1] == "gamma":
        tf_name[-1] = "weight"
    if tf_name[-1] == "beta":
        tf_name[-1] = "bias"

    # The SeparableConv1D TF layer contains two weights that are translated to PyTorch Conv1D here
    if tf_name[-1] == "pointwise_kernel" or tf_name[-1] == "depthwise_kernel":
        tf_name[-1] = tf_name[-1].replace("_kernel", ".weight")

    # Remove prefix if needed
    tf_name = ".".join(tf_name)
    if start_prefix_to_remove:
        tf_name = tf_name.replace(start_prefix_to_remove, "", 1)

    return tf_name, transpose


#####################
# PyTorch => TF 2.0 #
#####################


def load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path, tf_inputs=None, allow_missing_keys=False):
    """Load pytorch checkpoints in a TF 2.0 model"""
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    pt_path = os.path.abspath(pytorch_checkpoint_path)
    logger.info(f"Loading PyTorch weights from {pt_path}")

    pt_state_dict = torch.load(pt_path, map_location="cpu")
    logger.info(f"PyTorch checkpoint contains {sum(t.numel() for t in pt_state_dict.values()):,} parameters")

    return load_pytorch_weights_in_tf2_model(
        tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys
    )


def load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=None, allow_missing_keys=False):
    """Load pytorch checkpoints in a TF 2.0 model"""
    pt_state_dict = pt_model.state_dict()

    return load_pytorch_weights_in_tf2_model(
        tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys
    )


def load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=None, allow_missing_keys=False):
    """Load pytorch state_dict in a TF 2.0 model."""
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
        from tensorflow.python.keras import backend as K
    except ImportError:
        logger.error(
            "Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs

    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)  # Make sure model is built
    # Adapt state dict - TODO remove this and update the AWS weights files instead
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in pt_state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        pt_state_dict[new_key] = pt_state_dict.pop(old_key)

    # Make sure we are able to load PyTorch base models as well as derived models (with heads)
    # TF models always have a prefix, some of PyTorch models (base ones) don't
    start_prefix_to_remove = ""
    if not any(s.startswith(tf_model.base_model_prefix) for s in pt_state_dict.keys()):
        start_prefix_to_remove = tf_model.base_model_prefix + "."

    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    tf_loaded_numel = 0
    weight_value_tuples = []
    all_pytorch_weights = set(list(pt_state_dict.keys()))
    missing_keys = []
    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.name
        name, transpose = convert_tf_weight_name_to_pt_weight_name(
            sw_name, start_prefix_to_remove=start_prefix_to_remove, tf_weight_shape=symbolic_weight.shape
        )

        # Find associated numpy array in pytorch model state dict
        if name not in pt_state_dict:
            if allow_missing_keys:
                missing_keys.append(name)
                continue
            elif tf_model._keys_to_ignore_on_load_missing is not None:
                # authorized missing keys don't have to be loaded
                if any(re.search(pat, name) is not None for pat in tf_model._keys_to_ignore_on_load_missing):
                    continue
            raise AttributeError(f"{name} not found in PyTorch model")

        array = pt_state_dict[name].numpy()

        if transpose is TransposeType.CONV2D:
            # Conv2D weight:
            #    PT: (num_out_channel, num_in_channel, kernel[0], kernel[1])
            # -> TF: (kernel[0], kernel[1], num_in_channel, num_out_channel)
            array = numpy.transpose(array, axes=(2, 3, 1, 0))
        elif transpose is TransposeType.CONV1D:
            # Conv1D weight:
            #    PT: (num_out_channel, num_in_channel, kernel)
            # -> TF: (kernel, num_in_channel, num_out_channel)
            array = numpy.transpose(array, axes=(2, 1, 0))
        elif transpose is TransposeType.SIMPLE:
            array = numpy.transpose(array)

        if len(symbolic_weight.shape) < len(array.shape):
            array = numpy.squeeze(array)
        elif len(symbolic_weight.shape) > len(array.shape):
            array = numpy.expand_dims(array, axis=0)

        if list(symbolic_weight.shape) != list(array.shape):
            try:
                array = numpy.reshape(array, symbolic_weight.shape)
            except AssertionError as e:
                e.args += (symbolic_weight.shape, array.shape)
                raise e

        try:
            assert list(symbolic_weight.shape) == list(array.shape)
        except AssertionError as e:
            e.args += (symbolic_weight.shape, array.shape)
            raise e

        tf_loaded_numel += array.size
        # logger.warning(f"Initialize TF weight {symbolic_weight.name}")

        weight_value_tuples.append((symbolic_weight, array))
        all_pytorch_weights.discard(name)

    K.batch_set_value(weight_value_tuples)

    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)  # Make sure restore ops are run

    logger.info(f"Loaded {tf_loaded_numel:,} parameters in the TF 2.0 model.")

    unexpected_keys = list(all_pytorch_weights)

    if tf_model._keys_to_ignore_on_load_missing is not None:
        for pat in tf_model._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
    if tf_model._keys_to_ignore_on_load_unexpected is not None:
        for pat in tf_model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

    if len(unexpected_keys) > 0:
        logger.warning(
            "Some weights of the PyTorch model were not used when initializing the TF 2.0 model"
            f" {tf_model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing"
            f" {tf_model.__class__.__name__} from a PyTorch model trained on another task or with another architecture"
            " (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).\n- This IS"
            f" NOT expected if you are initializing {tf_model.__class__.__name__} from a PyTorch model that you expect"
            " to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a"
            " BertForSequenceClassification model)."
        )
    else:
        logger.warning(f"All PyTorch model weights were used when initializing {tf_model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights or buffers of the TF 2.0 model {tf_model.__class__.__name__} were not initialized from the"
            f" PyTorch model and are newly initialized: {missing_keys}\nYou should probably TRAIN this model on a"
            " down-stream task to be able to use it for predictions and inference."
        )
    else:
        logger.warning(
            f"All the weights of {tf_model.__class__.__name__} were initialized from the PyTorch model.\n"
            "If your task is similar to the task the model of the checkpoint was trained on, "
            f"you can already use {tf_model.__class__.__name__} for predictions without further training."
        )

    return tf_model


#####################
# TF 2.0 => PyTorch #
#####################


def load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path, tf_inputs=None, allow_missing_keys=False):
    """
    Load TF 2.0 HDF5 checkpoint in a PyTorch model We use HDF5 to easily do transfer learning (see
    https://github.com/tensorflow/tensorflow/blob/ee16fcac960ae660e0e4496658a366e2f745e1f0/tensorflow/python/keras/engine/network.py#L1352-L1357).
    """
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    import transformers

    from .modeling_tf_utils import load_tf_weights

    logger.info(f"Loading TensorFlow weights from {tf_checkpoint_path}")

    # Instantiate and load the associated TF 2.0 model
    tf_model_class_name = "TF" + pt_model.__class__.__name__  # Add "TF" at the beginning
    tf_model_class = getattr(transformers, tf_model_class_name)
    tf_model = tf_model_class(pt_model.config)

    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs

    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)  # Make sure model is built

    load_tf_weights(tf_model, tf_checkpoint_path)

    return load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=allow_missing_keys)


def load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=False):
    """Load TF 2.0 model in a pytorch model"""
    weights = tf_model.weights

    return load_tf2_weights_in_pytorch_model(pt_model, weights, allow_missing_keys=allow_missing_keys)


def load_tf2_weights_in_pytorch_model(pt_model, tf_weights, allow_missing_keys=False):
    """Load TF2.0 symbolic weights in a PyTorch model"""
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    new_pt_params_dict = {}
    current_pt_params_dict = dict(pt_model.named_parameters())

    # Make sure we are able to load PyTorch base models as well as derived models (with heads)
    # TF models always have a prefix, some of PyTorch models (base ones) don't
    start_prefix_to_remove = ""
    if not any(s.startswith(pt_model.base_model_prefix) for s in current_pt_params_dict.keys()):
        start_prefix_to_remove = pt_model.base_model_prefix + "."

    # Build a map from potential PyTorch weight names to TF 2.0 Variables
    tf_weights_map = {}
    for tf_weight in tf_weights:
        pt_name, transpose = convert_tf_weight_name_to_pt_weight_name(
            tf_weight.name, start_prefix_to_remove=start_prefix_to_remove, tf_weight_shape=tf_weight.shape
        )
        tf_weights_map[pt_name] = (tf_weight.numpy(), transpose)

    all_tf_weights = set(list(tf_weights_map.keys()))
    loaded_pt_weights_data_ptr = {}
    missing_keys_pt = []
    for pt_weight_name, pt_weight in current_pt_params_dict.items():
        # Handle PyTorch shared weight ()not duplicated in TF 2.0
        if pt_weight.data_ptr() in loaded_pt_weights_data_ptr:
            new_pt_params_dict[pt_weight_name] = loaded_pt_weights_data_ptr[pt_weight.data_ptr()]
            continue

        # Find associated numpy array in pytorch model state dict
        if pt_weight_name not in tf_weights_map:
            if allow_missing_keys:
                missing_keys_pt.append(pt_weight_name)
                continue

            raise AttributeError(f"{pt_weight_name} not found in TF 2.0 model")

        array, transpose = tf_weights_map[pt_weight_name]

        if transpose is TransposeType.CONV2D:
            # Conv2D weight:
            #    TF: (kernel[0], kernel[1], num_in_channel, num_out_channel)
            # -> PT: (num_out_channel, num_in_channel, kernel[0], kernel[1])
            array = numpy.transpose(array, axes=(3, 2, 0, 1))
        elif transpose is TransposeType.CONV1D:
            # Conv1D weight:
            #    TF: (kernel, num_in_channel, num_out_channel)
            # -> PT: (num_out_channel, num_in_channel, kernel)
            array = numpy.transpose(array, axes=(2, 1, 0))
        elif transpose is TransposeType.SIMPLE:
            array = numpy.transpose(array)

        if len(pt_weight.shape) < len(array.shape):
            array = numpy.squeeze(array)
        elif len(pt_weight.shape) > len(array.shape):
            array = numpy.expand_dims(array, axis=0)

        if list(pt_weight.shape) != list(array.shape):
            try:
                array = numpy.reshape(array, pt_weight.shape)
            except AssertionError as e:
                e.args += (pt_weight.shape, array.shape)
                raise e

        try:
            assert list(pt_weight.shape) == list(array.shape)
        except AssertionError as e:
            e.args += (pt_weight.shape, array.shape)
            raise e

        # logger.warning(f"Initialize PyTorch weight {pt_weight_name}")
        # Make sure we have a proper numpy array
        if numpy.isscalar(array):
            array = numpy.array(array)
        new_pt_params_dict[pt_weight_name] = torch.from_numpy(array)
        loaded_pt_weights_data_ptr[pt_weight.data_ptr()] = torch.from_numpy(array)
        all_tf_weights.discard(pt_weight_name)

    missing_keys, unexpected_keys = pt_model.load_state_dict(new_pt_params_dict, strict=False)
    missing_keys += missing_keys_pt

    # Some models may have keys that are not in the state by design, removing them before needlessly warning
    # the user.
    if pt_model._keys_to_ignore_on_load_missing is not None:
        for pat in pt_model._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

    if pt_model._keys_to_ignore_on_load_unexpected is not None:
        for pat in pt_model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

    if len(unexpected_keys) > 0:
        logger.warning(
            "Some weights of the TF 2.0 model were not used when initializing the PyTorch model"
            f" {pt_model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing"
            f" {pt_model.__class__.__name__} from a TF 2.0 model trained on another task or with another architecture"
            " (e.g. initializing a BertForSequenceClassification model from a TFBertForPreTraining model).\n- This IS"
            f" NOT expected if you are initializing {pt_model.__class__.__name__} from a TF 2.0 model that you expect"
            " to be exactly identical (e.g. initializing a BertForSequenceClassification model from a"
            " TFBertForSequenceClassification model)."
        )
    else:
        logger.warning(f"All TF 2.0 model weights were used when initializing {pt_model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {pt_model.__class__.__name__} were not initialized from the TF 2.0 model and are newly"
            f" initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to"
            " use it for predictions and inference."
        )
    else:
        logger.warning(
            f"All the weights of {pt_model.__class__.__name__} were initialized from the TF 2.0 model.\n"
            "If your task is similar to the task the model of the checkpoint was trained on, "
            f"you can already use {pt_model.__class__.__name__} for predictions without further training."
        )

    logger.info(f"Weights or buffers not loaded from TF 2.0 model: {all_tf_weights}")

    return pt_model
