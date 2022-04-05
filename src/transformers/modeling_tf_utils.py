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
"""TF general model utils."""

import functools
import inspect
import os
import pickle
import re
import warnings
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.keras.saving import hdf5_format

from huggingface_hub import Repository, list_repo_files
from requests import HTTPError

from .activations_tf import get_tf_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation_tf_utils import TFGenerationMixin
from .tf_utils import shape_list
from .tokenization_utils_base import BatchEncoding
from .utils import (
    DUMMY_INPUTS,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    TF2_WEIGHTS_NAME,
    WEIGHTS_NAME,
    EntryNotFoundError,
    ModelOutput,
    PushToHubMixin,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    cached_path,
    copy_func,
    find_labels,
    has_file,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
    logging,
)


logger = logging.get_logger(__name__)
tf_logger = tf.get_logger()

TFModelInputType = Union[
    List[tf.Tensor],
    List[np.ndarray],
    List[KerasTensor],
    Dict[str, tf.Tensor],
    Dict[str, np.ndarray],
    Dict[str, KerasTensor],
    tf.Tensor,
    np.ndarray,
    KerasTensor,
]


def dummy_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


class TFModelUtilsMixin:
    """
    A few utilities for `tf.keras.Model`, to be used as a mixin.
    """

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Get the number of (optionally, trainable) parameters in the model.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

        Returns:
            `int`: The number of parameters.
        """
        if only_trainable:
            return int(sum(np.prod(w.shape.as_list()) for w in self.trainable_variables))
        else:
            return self.count_params()


def keras_serializable(cls):
    """
    Decorate a Keras Layer class to support Keras serialization.

    This is done by:

    1. Adding a `transformers_config` dict to the Keras config dictionary in `get_config` (called by Keras at
       serialization time.
    2. Wrapping `__init__` to accept that `transformers_config` dict (passed by Keras at deserialization time) and
       convert it to a config object for the actual layer initializer.
    3. Registering the class as a custom object in Keras (if the Tensorflow version supports this), so that it does not
       need to be supplied in `custom_objects` in the call to `tf.keras.models.load_model`.

    Args:
        cls (a `tf.keras.layers.Layers subclass`):
            Typically a `TF.MainLayer` class in this project, in general must accept a `config` argument to its
            initializer.

    Returns:
        The same class object, with modifications for Keras deserialization.
    """
    initializer = cls.__init__

    config_class = getattr(cls, "config_class", None)
    if config_class is None:
        raise AttributeError("Must set `config_class` to use @keras_serializable")

    @functools.wraps(initializer)
    def wrapped_init(self, *args, **kwargs):
        config = args[0] if args and isinstance(args[0], PretrainedConfig) else kwargs.pop("config", None)

        if isinstance(config, dict):
            config = config_class.from_dict(config)
            initializer(self, config, *args, **kwargs)
        elif isinstance(config, PretrainedConfig):
            if len(args) > 0:
                initializer(self, *args, **kwargs)
            else:
                initializer(self, config, *args, **kwargs)
        else:
            raise ValueError("Must pass either `config` (PretrainedConfig) or `config` (dict)")

        self._config = config
        self._kwargs = kwargs

    cls.__init__ = wrapped_init

    if not hasattr(cls, "get_config"):
        raise TypeError("Only use @keras_serializable on tf.keras.layers.Layer subclasses")
    if hasattr(cls.get_config, "_is_default"):

        def get_config(self):
            cfg = super(cls, self).get_config()
            cfg["config"] = self._config.to_dict()
            cfg.update(self._kwargs)
            return cfg

        cls.get_config = get_config

    cls._keras_serializable = True
    if hasattr(tf.keras.utils, "register_keras_serializable"):
        cls = tf.keras.utils.register_keras_serializable()(cls)
    return cls


class TFCausalLanguageModelingLoss:
    """
    Loss function suitable for causal language modeling (CLM), that is, the task of guessing the next token.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """

    def hf_compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # make sure only labels that are not equal to -100 affect the loss
        active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
        reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
        labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
        return loss_fn(labels, reduced_logits)


class TFQuestionAnsweringLoss:
    """
    Loss function suitable for question answering.
    """

    def hf_compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        start_loss = loss_fn(labels["start_position"], logits[0])
        end_loss = loss_fn(labels["end_position"], logits[1])

        return (start_loss + end_loss) / 2.0


class TFTokenClassificationLoss:
    """
    Loss function suitable for token classification.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """

    def hf_compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # make sure only labels that are not equal to -100
        # are taken into account as loss
        if tf.math.reduce_any(labels == -1):
            tf.print("Using `-1` to mask the loss for the token is deprecated. Please use `-100` instead.")
            active_loss = tf.reshape(labels, (-1,)) != -1
        else:
            active_loss = tf.reshape(labels, (-1,)) != -100
        reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
        labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)

        return loss_fn(labels, reduced_logits)


class TFSequenceClassificationLoss:
    """
    Loss function suitable for sequence classification.
    """

    def hf_compute_loss(self, labels, logits):
        if len(shape_list(logits)) == 1 or shape_list(logits)[1] == 1:
            loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        else:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )

        return loss_fn(labels, logits)


class TFMultipleChoiceLoss:
    """Loss function suitable for multiple choice tasks."""

    def hf_compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        return loss_fn(labels, logits)


class TFMaskedLanguageModelingLoss(TFCausalLanguageModelingLoss):
    """
    Loss function suitable for masked language modeling (MLM), that is, the task of guessing the masked tokens.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """


class TFNextSentencePredictionLoss:
    """
    Loss function suitable for next sentence prediction (NSP), that is, the task of guessing the next sentence.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """

    def hf_compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # make sure only labels that are not equal to -100
        # are taken into account as loss
        next_sentence_active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
        next_sentence_reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, 2)), next_sentence_active_loss)
        next_sentence_label = tf.boolean_mask(tf.reshape(labels, (-1,)), next_sentence_active_loss)

        return loss_fn(next_sentence_label, next_sentence_reduced_logits)


def booleans_processing(config, **kwargs):
    """
    Process the input booleans of each model in order to be sure they are compliant with the execution mode (eager or
    graph)

    Args:
        config ([`PretrainedConfig`]):
            The config of the running model.
        **kwargs:
            The boolean parameters

    Returns:
        A dictionary with the proper values for each boolean
    """
    final_booleans = {}

    if tf.executing_eagerly():
        # Pure conv models (such as ConvNext) do not have `output_attentions`. If the signature has
        # `output_attentions`, it will be present here in `kwargs`, even if unset (in that case, as `None`)
        if "output_attentions" in kwargs:
            final_booleans["output_attentions"] = (
                kwargs["output_attentions"] if kwargs["output_attentions"] is not None else config.output_attentions
            )
        final_booleans["output_hidden_states"] = (
            kwargs["output_hidden_states"]
            if kwargs["output_hidden_states"] is not None
            else config.output_hidden_states
        )
        final_booleans["return_dict"] = (
            kwargs["return_dict"] if kwargs["return_dict"] is not None else config.return_dict
        )

        if "use_cache" in kwargs:
            final_booleans["use_cache"] = (
                kwargs["use_cache"] if kwargs["use_cache"] is not None else getattr(config, "use_cache", None)
            )
    else:
        # Pure conv models (such as ConvNext) do not have `output_attentions`. If the signature has
        # `output_attentions`, it will be present here in `kwargs`, even if unset (in that case, as `None`)
        if "output_attentions" in kwargs:
            final_booleans["output_attentions"] = config.output_attentions
        final_booleans["output_hidden_states"] = config.output_hidden_states

        if kwargs.get("return_dict", None) not in (None, True):
            tf_logger.warning(
                "The parameter `return_dict` cannot be set in graph mode and will always be set to `True`."
            )
        final_booleans["return_dict"] = True

        if "use_cache" in kwargs:
            final_booleans["use_cache"] = getattr(config, "use_cache", None)

    return final_booleans


def unpack_inputs(func):
    """
    Decorator that processes the inputs to a Keras layer, passing them to the layer as keyword arguments. This enables
    downstream use of the inputs by their variable name, even if they arrive packed as a dictionary in the first input
    (common case in Keras).

    Args:
        func (`callable`):
            The callable function of the TensorFlow model.

    Returns:
        A callable that wraps the original `func` with the behavior described above.
    """

    original_signature = inspect.signature(func)

    @functools.wraps(func)
    def run_call_with_unpacked_inputs(self, *args, **kwargs):
        # isolates the actual `**kwargs` for the decorated function
        kwargs_call = {key: val for key, val in kwargs.items() if key not in dict(original_signature.parameters)}
        fn_args_and_kwargs = {key: val for key, val in kwargs.items() if key not in kwargs_call}
        fn_args_and_kwargs.update({"kwargs_call": kwargs_call})

        # move any arg into kwargs, if they exist
        fn_args_and_kwargs.update(dict(zip(func.__code__.co_varnames[1:], args)))

        # process the inputs and call the wrapped function
        main_input_name = getattr(self, "main_input_name", func.__code__.co_varnames[1])
        main_input = fn_args_and_kwargs.pop(main_input_name, None)
        unpacked_inputs = input_processing(func, self.config, main_input, **fn_args_and_kwargs)
        return func(self, **unpacked_inputs)

    # Keras enforces the first layer argument to be passed, and checks it through `inspect.getfullargspec()`. This
    # function does not follow wrapper chains (i.e. ignores `functools.wraps()`), meaning that without the line below
    # Keras would attempt to check the first argument against the literal signature of the wrapper.
    run_call_with_unpacked_inputs.__signature__ = original_signature

    return run_call_with_unpacked_inputs


def input_processing(func, config, input_ids, **kwargs):
    """
    Process the input of each TensorFlow model including the booleans. In case of a list of symbolic inputs, each input
    has to be named accordingly to the parameters name, i.e. `input_ids = tf.keras.Input(shape=(128,), dtype='int32',
    name="input_ids")` otherwise the order of the tensors will not be guaranteed during the training.

    Args:
        func (`callable`):
            The callable function of the TensorFlow model.
        config ([`PretrainedConfig`]):
            The config of the running model.
        **kwargs:
            The inputs of the model.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    signature = dict(inspect.signature(func).parameters)
    has_kwargs = bool(signature.pop("kwargs", None))
    signature.pop("self", None)
    parameter_names = list(signature.keys())
    output = {}
    allowed_types = (tf.Tensor, bool, int, ModelOutput, tuple, list, dict, np.ndarray, KerasTensor)

    if "inputs" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.",
            FutureWarning,
        )

        output["input_ids"] = kwargs["kwargs_call"].pop("inputs")

    if "decoder_cached_states" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
            FutureWarning,
        )
        output["past_key_values"] = kwargs["kwargs_call"].pop("decoder_cached_states")

    if "past" in kwargs["kwargs_call"] and "past_key_values" in parameter_names:
        warnings.warn(
            "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
            FutureWarning,
        )
        kwargs["past_key_values"] = kwargs["kwargs_call"].pop("past")
    elif "past_key_values" in kwargs["kwargs_call"] and "past" in parameter_names:
        kwargs["past"] = kwargs["kwargs_call"].pop("past_key_values")

    if has_kwargs:
        output["kwargs"] = kwargs.pop("kwargs_call", {})
    else:
        if len(kwargs["kwargs_call"]) > 0:
            raise ValueError(
                f"The following keyword arguments are not supported by this model: {list(kwargs['kwargs_call'].keys())}."
            )
        kwargs.pop("kwargs_call")

    for k, v in kwargs.items():
        if isinstance(v, allowed_types) or v is None:
            output[k] = v
        else:
            raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")

    if isinstance(input_ids, (tuple, list)):
        for i, input in enumerate(input_ids):
            # EagerTensors don't allow to use the .name property so we check for a real Tensor
            if type(input) == tf.Tensor:
                # Tensor names have always the pattern `name:id` then we check only the
                # `name` part
                tensor_name = input.name.split(":")[0]

                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    output[parameter_names[i]] = input
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                raise ValueError(
                    f"Data of type {type(input)} is not allowed only {allowed_types} is accepted for {parameter_names[i]}."
                )
    elif isinstance(input_ids, (dict, BatchEncoding)):
        if "inputs" in input_ids:
            warnings.warn(
                "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.",
                FutureWarning,
            )

            output["input_ids"] = input_ids.pop("inputs")

        if "decoder_cached_states" in input_ids:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            output["past_key_values"] = input_ids.pop("decoder_cached_states")

        for k, v in dict(input_ids).items():
            if isinstance(v, allowed_types) or v is None:
                output[k] = v
            elif k not in parameter_names and "args" not in parameter_names:
                logger.warning(
                    f"The parameter {k} does not belongs to the parameter list {parameter_names} and will be ignored."
                )
                continue
            else:
                raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    else:
        if isinstance(input_ids, (tf.Tensor, KerasTensor)) or input_ids is None:
            output[parameter_names[0]] = input_ids
        else:
            raise ValueError(
                f"Data of type {type(input_ids)} is not allowed only {allowed_types} is accepted for {parameter_names[0]}."
            )

    # Populates any unspecified argument with their default value, according to the signature.
    for name in parameter_names:
        if name not in list(output.keys()) and name != "args":
            output[name] = kwargs.pop(name, signature[name].default)

    # When creating a SavedModel TF calls the method with LayerCall.__call__(args, **kwargs)
    # So to respect the proper output we have to add this exception
    if "args" in output:
        if output["args"] is not None and type(output["args"]) == tf.Tensor:
            tensor_name = output["args"].name.split(":")[0]
            output[tensor_name] = output["args"]
        else:
            # `args` in this case is always the first parameter, then `input_ids`
            output["input_ids"] = output["args"]

        del output["args"]

    if "kwargs" in output:
        del output["kwargs"]

    boolean_dict = {
        k: v
        for k, v in output.items()
        if k in ["return_dict", "output_attentions", "output_hidden_states", "use_cache"]
    }

    output.update(
        booleans_processing(
            config=config,
            **boolean_dict,
        )
    )

    return output


def load_tf_weights(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    """
    Detect missing and unexpected layers and load the TF weights accordingly to their names and shapes.

    Args:
        model (`tf.keras.models.Model`):
            The model to load the weights into.
        resolved_archive_file (`str`):
            The location of the H5 file.
        ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
            Whether or not to ignore weights with shapes that don't match between the checkpoint of the model.

    Returns:
        Three lists, one for the missing layers, another one for the unexpected layers, and a last one for the
        mismatched layers.
    """
    missing_layers = []
    unexpected_layers = []
    mismatched_layers = []

    # Read the H5 file
    with h5py.File(resolved_archive_file, "r") as f:
        # Retrieve the name of each layer from the H5 file
        saved_h5_model_layers_name = set(hdf5_format.load_attributes_from_hdf5_group(f, "layer_names"))

        # Find the missing layers from the high level list of layers
        missing_layers = list(set([layer.name for layer in model.layers]) - saved_h5_model_layers_name)

        # Find the unexpected layers from the high level list of layers
        unexpected_layers = list(saved_h5_model_layers_name - set([layer.name for layer in model.layers]))
        saved_weight_names_set = set()
        symbolic_weights_names = set()
        weight_value_tuples = []

        # Compute missing and unexpected sub layers
        # Store the weights in list of tuples that looks like [(weight_object, value_of_weight),...]
        for layer in model.layers:
            # if layer_name from the H5 file belongs to the layers from the instantiated model
            if layer.name in saved_h5_model_layers_name:
                # Get the H5 layer object from its name
                h5_layer_object = f[layer.name]
                # Get all the weights as a list from the layer object
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                saved_weights = {}

                # Create a dict from the H5 saved model that looks like {"weight_name": weight_value}
                # And a set with only the names
                for weight_name in hdf5_format.load_attributes_from_hdf5_group(h5_layer_object, "weight_names"):
                    # TF names always start with the model name so we ignore it
                    name = "/".join(weight_name.split("/")[1:])

                    if _prefix is not None:
                        name = _prefix + "/" + name

                    saved_weights[name] = np.asarray(h5_layer_object[weight_name])

                    # Add the updated name to the final list for computing missing/unexpected values
                    saved_weight_names_set.add(name)

                # Loop over each weights from the instantiated model and compare with the weights from the H5 file
                for symbolic_weight in symbolic_weights:
                    # TF names always start with the model name so we ignore it
                    if _prefix is not None:
                        delimeter = len(_prefix.split("/"))
                        symbolic_weight_name = "/".join(
                            symbolic_weight.name.split("/")[:delimeter]
                            + symbolic_weight.name.split("/")[delimeter + 1 :]
                        )
                    else:
                        symbolic_weight_name = "/".join(symbolic_weight.name.split("/")[1:])

                    # here we check if the current weight is among the weights from the H5 file
                    # If yes, get the weight_value of the corresponding weight from the H5 file
                    # If not, make the value to None
                    saved_weight_value = saved_weights.get(symbolic_weight_name, None)

                    # Add the updated name to the final list for computing missing/unexpected values
                    symbolic_weights_names.add(symbolic_weight_name)

                    # If the current weight is found
                    if saved_weight_value is not None:
                        # Check if the shape of the current weight and the one from the H5 file are different
                        if K.int_shape(symbolic_weight) != saved_weight_value.shape:
                            # If yes we reshape the weight from the H5 file accordingly to the current weight
                            # If the two shapes are not compatible we raise an issue
                            try:
                                array = np.reshape(saved_weight_value, K.int_shape(symbolic_weight))
                            except ValueError as e:
                                if ignore_mismatched_sizes:
                                    mismatched_layers.append(
                                        (symbolic_weight_name, saved_weight_value.shape, K.int_shape(symbolic_weight))
                                    )
                                    continue
                                else:
                                    raise e
                        else:
                            array = saved_weight_value

                        # We create the tuple that will be loaded and add it to the final list
                        weight_value_tuples.append((symbolic_weight, array))

    # Load all the weights
    K.batch_set_value(weight_value_tuples)

    # Compute the missing and unexpected layers
    missing_layers.extend(list(symbolic_weights_names - saved_weight_names_set))
    unexpected_layers.extend(list(saved_weight_names_set - symbolic_weights_names))

    return missing_layers, unexpected_layers, mismatched_layers


def init_copy_embeddings(old_embeddings, new_num_tokens):
    r"""
    This function aims to reduce the embeddings in case new_num_tokens < old_num_tokens or to pad with -1 in case
    new_num_tokens > old_num_tokens. A mask is also computed in order to know which weight in the embeddings should be
    kept or not. Example:

        - if new_num_tokens=5 and old_num_tokens=4 and old_embeddings=[w1,w2,w3,w4]

            -  mask=[True,True,True,True,False] and current_weights=[w1,w2,w3,w4,-1]
        - if new_num_tokens=4 and old_num_tokens=5 and old_embeddings=[w1,w2,w3,w4,w5]

            - mask=[True,True,True,True] and current_weights=[w1,w2,w3,w4]
    """
    old_num_tokens, old_embedding_dim = shape_list(old_embeddings)
    size_diff = new_num_tokens - old_num_tokens

    # initialize new embeddings
    # Copy token embeddings from the previous ones
    if tf.math.greater(size_diff, 0):
        # if the new size is greater than the old one, we extend the current embeddings with a padding until getting new size
        # and we create a mask to properly identify the padded values and be replaced by the values of the newly created
        # embeddings
        current_weights = tf.pad(
            old_embeddings.value(), tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=-1
        )
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        mask = tf.fill(tf.convert_to_tensor([num_tokens_to_copy, 1]), True)
        mask = tf.pad(mask, tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=False)
    else:
        # if the new size if lower than the old one, we take the current embeddings until the new size
        current_weights = tf.slice(
            old_embeddings.value(),
            tf.convert_to_tensor([0, 0]),
            tf.convert_to_tensor([new_num_tokens, old_embedding_dim]),
        )
        mask = tf.fill(tf.convert_to_tensor([new_num_tokens, 1]), True)

    return mask, current_weights


class TFPreTrainedModel(tf.keras.Model, TFModelUtilsMixin, TFGenerationMixin, PushToHubMixin):
    r"""
    Base class for all TF models.

    [`TFPreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings,
        - prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    """
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    _auto_class = None
    _using_dummy_loss = None

    # a list of re pattern of tensor names to ignore from the model when loading the model weights
    # (and avoid unnecessary warnings).
    _keys_to_ignore_on_load_missing = None
    # a list of re pattern of tensor names to ignore from the weights when loading the model weights
    # (and avoid unnecessary warnings).
    _keys_to_ignore_on_load_unexpected = None
    _requires_load_weight_prefix = False

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        return {
            "input_ids": tf.constant(DUMMY_INPUTS),
        }

    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a TensorFlow model.
        """
        return "tf"

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # Save config and origin of the pretrained weights if given in model
        self.config = config
        self.name_or_path = config.name_or_path

    def get_config(self):
        return self.config.to_dict()

    @classmethod
    def from_config(cls, config, **kwargs):
        if isinstance(config, PretrainedConfig):
            return cls._from_config(config, **kwargs)
        return cls._from_config(cls.config_class.from_dict(config, **kwargs))

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.
        """
        return cls(config, **kwargs)

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
                "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
                "token_type_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids"),
            }
        ]
    )
    def serving(self, inputs):
        """
        Method used for serving the model.

        Args:
            inputs (`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """
        output = self.call(inputs)

        return self.serving_output(output)

    def serving_output(output):
        """
        Prepare the output of the saved model. Each model must implement this function.

        Args:
            output ([`TFBaseModelOutput`]):
                The output returned by the model.
        """
        raise NotImplementedError

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        """
        Returns the model's input embeddings layer.

        Returns:
            `tf.Variable`: The embeddings layer mapping vocabulary to hidden states.
        """
        main_layer = getattr(self, self.base_model_prefix, self)

        if main_layer is not self:
            return main_layer.get_input_embeddings()
        else:
            raise NotImplementedError

    def _save_checkpoint(self, checkpoint_dir, epoch):
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        # We avoid tf.train.checkpoint or saving weights in TF format, even though that includes optimizer
        # state for us, because it requires special handling for objects like custom losses, which we use
        # internally and which users are likely to use too
        weights_path = os.path.join(checkpoint_dir, "weights.h5")
        self.save_weights(weights_path)
        extra_data = {"epoch": epoch, "optimizer_state": self.optimizer.get_weights()}
        extra_data_path = os.path.join(checkpoint_dir, "extra_data.pickle")
        with open(extra_data_path, "wb") as f:
            pickle.dump(extra_data, f)

    def load_repo_checkpoint(self, repo_path_or_name):
        """
        Loads a saved checkpoint (model weights and optimizer state) from a repo. Returns the current epoch count when
        the checkpoint was made.

        Args:
            repo_path_or_name (`str`):
                Can either be a repository name for your {object} in the Hub or a path to a local folder (in which case
                the repository will have the name of that local folder).

        Returns:
            `dict`: A dictionary of extra metadata from the checkpoint, most commonly an "epoch" count.
        """
        if getattr(self, "optimizer", None) is None:
            raise RuntimeError(
                "Checkpoint loading failed as no optimizer is attached to the model. "
                "This is most likely caused by the model not being compiled."
            )
        if not os.path.isdir(repo_path_or_name):
            # If this isn't a local path, check that the remote repo exists and has a checkpoint in it
            repo_files = list_repo_files(repo_path_or_name)
            for file in ("checkpoint/weights.h5", "checkpoint/extra_data.pickle"):
                if file not in repo_files:
                    raise FileNotFoundError(f"Repo {repo_path_or_name} does not contain checkpoint file {file}!")
            if "/" not in repo_path_or_name:
                model_id = repo_path_or_name
                repo_path_or_name = self.get_full_repo_name(repo_path_or_name)
            else:
                model_id = repo_path_or_name.split("/")[-1]
            repo = Repository(model_id, clone_from=f"https://huggingface.co/{repo_path_or_name}")
            local_dir = repo.local_dir
        else:
            local_dir = repo_path_or_name

        # Now make sure the repo actually has a checkpoint in it.
        checkpoint_dir = os.path.join(local_dir, "checkpoint")
        weights_file = os.path.join(checkpoint_dir, "weights.h5")
        if not os.path.isfile(weights_file):
            raise FileNotFoundError(f"Could not find checkpoint file weights.h5 in repo {repo_path_or_name}!")
        extra_data_file = os.path.join(checkpoint_dir, "extra_data.pickle")
        if not os.path.isfile(extra_data_file):
            raise FileNotFoundError(f"Could not find checkpoint file extra_data.pickle in repo {repo_path_or_name}!")

        # Assuming the repo is real and we got a checkpoint, load the weights and the optimizer state into the model.
        # The optimizer state includes the iteration count, so learning rate schedules should resume as normal too.
        self.load_weights(weights_file)
        with open(extra_data_file, "rb") as f:
            extra_data = pickle.load(f)
        self.optimizer.set_weights(extra_data["optimizer_state"])

        # Finally, return the epoch number from the checkpoint. This isn't a property of the model, so we can't
        # set it directly, but the user can pass it to fit().
        return {"epoch": extra_data["epoch"]}

    def compile(
        self,
        optimizer="rmsprop",
        loss="passthrough",
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs
    ):
        """
        This is a thin wrapper that sets the model's loss output head as the loss if the user does not specify a loss
        function themselves.
        """
        if loss == "passthrough":
            if metrics is not None:
                raise ValueError(
                    "Passing metrics as a dict is not supported when using the internal loss! "
                    "Please either compile the model with a loss, or remove the metrics argument. "
                    "Note that advanced metrics using the `KerasMetricCallback` can still be used with the internal "
                    "loss."
                )
            logger.warning(
                "No loss specified in compile() - the model's internal loss computation will be used as the "
                "loss. Don't panic - this is a common way to train TensorFlow models in Transformers! "
                "To disable this behaviour, please pass a loss argument, or explicitly pass "
                "`loss=None` if you do not want your model to compute a loss."
            )
            loss = dummy_loss
            self._using_dummy_loss = True
        else:
            self._using_dummy_loss = False
        parent_args = list(inspect.signature(tf.keras.Model.compile).parameters.keys())
        if "steps_per_execution" in parent_args:
            super().compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                weighted_metrics=weighted_metrics,
                run_eagerly=run_eagerly,
                steps_per_execution=steps_per_execution,
                **kwargs,
            )
        else:
            super().compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                weighted_metrics=weighted_metrics,
                run_eagerly=run_eagerly,
                experimental_steps_per_execution=steps_per_execution,
                **kwargs,
            )

    def compute_loss(self, *args, **kwargs):
        if hasattr(tf.keras.Model, "compute_loss"):
            # This will be true in TF 2.8 or greater
            return super().compute_loss(*args, **kwargs)
        else:
            warnings.warn(
                "The old compute_loss method is deprecated as it conflicts with the Keras compute_loss "
                "method added in TF 2.8. If you want the original HF compute_loss, please call "
                "hf_compute_loss() instead. From TF versions >= 2.8, or Transformers versions >= 5, "
                "calling compute_loss() will get the Keras method instead.",
                FutureWarning,
            )
            return self.hf_compute_loss(*args, **kwargs)

    def train_step(self, data):
        """
        A modification of Keras's default `train_step` that cleans up the printed metrics when we use a dummy loss. If
        a user specifies a loss at model compile time, this function behaves as the original Keras `train_step`.

        When the model is compiled without specifying the loss, our overridden compile function can set a simple dummy
        loss that just reads the loss output head of the model. When using this dummy loss, inputs can be passed either
        as keys in the input dictionary, or as normal Keras labels.
        """

        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        if not self._using_dummy_loss:
            data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # When using a dummy loss, we ensure that separate labels are copied to the correct model arguments,
        # if those keys are not already present in the input dict
        if self._using_dummy_loss and y is not None:
            arg_names = list(dict(inspect.signature(self.call).parameters).keys())
            label_kwargs = find_labels(self.__class__)
            # If y is a tensor and the model only has one label-like input, map y to that input
            if len(label_kwargs) == 1 and isinstance(y, tf.Tensor):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                label_kwarg = next(iter(label_kwargs))
                if label_kwarg not in x:
                    x[label_kwarg] = y
            # Otherwise, copy keys from y to x as long as they weren't already present in x
            elif isinstance(y, dict):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                for key, val in y.items():
                    if key in arg_names and key not in x:
                        x[key] = val

        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            if self._using_dummy_loss:
                loss = self.compiled_loss(y_pred.loss, y_pred.loss, sample_weight, regularization_losses=self.losses)
            else:
                loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        # When using the dummy_loss we know metrics are not present, so we can skip a lot of this
        if self._using_dummy_loss:
            self.compiled_metrics.update_state(y_pred.loss, y_pred.loss, sample_weight)
        else:
            self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        # These next two lines are also not in the base method - they correct the displayed metrics
        # when we're using a dummy loss, to avoid a bogus "loss_loss" value being shown.
        if "loss" in return_metrics and "loss_loss" in return_metrics:
            del return_metrics["loss_loss"]
        return return_metrics

    def test_step(self, data):
        """
        A modification of Keras's default `test_step` that cleans up the printed metrics when we use a dummy loss. If a
        user specifies a loss at model compile time, this function behaves as the original Keras `test_step`.

        When the model is compiled without specifying the loss, our overridden compile function can set a simple dummy
        loss that just reads the loss output head of the model. When using this dummy loss, inputs can be passed either
        as keys in the input dictionary, or as normal Keras labels.
        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        if not self._using_dummy_loss:
            data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # When using a dummy loss, we ensure that separate labels are copied to the correct model arguments,
        # if those keys are not already present in the input dict
        if self._using_dummy_loss and y is not None:
            arg_names = list(dict(inspect.signature(self.call).parameters).keys())
            label_kwargs = find_labels(self.__class__)
            # If y is a tensor and the model only has one label-like input, map y to that input
            if len(label_kwargs) == 1 and isinstance(y, tf.Tensor):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                label_kwarg = next(iter(label_kwargs))
                if label_kwarg not in x:
                    x[label_kwarg] = y
            # Otherwise, copy keys from y to x as long as they weren't already present in x
            elif isinstance(y, dict):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                for key, val in y.items():
                    if key in arg_names and key not in x:
                        x[key] = val

        # Run forward pass.
        y_pred = self(x, training=False)
        if self._using_dummy_loss:
            self.compiled_loss(y_pred.loss, y_pred.loss, sample_weight, regularization_losses=self.losses)
        else:
            self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        # When using the dummy_loss we know metrics are not present, so we can skip a lot of this
        if self._using_dummy_loss:
            self.compiled_metrics.update_state(y_pred.loss, y_pred.loss, sample_weight)
        else:
            self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        # These next two lines are also not in the base method - they correct the displayed metrics
        # when we're using a dummy loss, to avoid a bogus "loss_loss" value being shown.
        if "loss" in return_metrics and "loss_loss" in return_metrics:
            del return_metrics["loss_loss"]
        return return_metrics

    def create_model_card(
        self,
        output_dir,
        model_name: str,
        language: Optional[str] = None,
        license: Optional[str] = None,
        tags: Optional[str] = None,
        finetuned_from: Optional[str] = None,
        tasks: Optional[str] = None,
        dataset_tags: Optional[Union[str, List[str]]] = None,
        dataset: Optional[Union[str, List[str]]] = None,
        dataset_args: Optional[Union[str, List[str]]] = None,
    ):
        # Avoids a circular import by doing this when necessary.
        from .modelcard import TrainingSummary

        training_summary = TrainingSummary.from_keras(
            self,
            keras_history=self.history,
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset_tags=dataset_tags,
            dataset=dataset,
            dataset_args=dataset_args,
        )
        model_card = training_summary.to_model_card()
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(model_card)

    def set_input_embeddings(self, value):
        """
        Set model's input embeddings

        Args:
            value (`tf.Variable`):
                The new weights mapping hidden states to vocabulary.
        """
        main_layer = getattr(self, self.base_model_prefix)

        if main_layer is None:
            raise NotImplementedError("The model does not implements the base_model_prefix attribute.")

        try:
            main_layer.set_input_embeddings(value)
        except AttributeError:
            logger.info("Building the model")
            self(self.dummy_inputs)
            main_layer.set_input_embeddings(value)

    def get_output_embeddings(self) -> Union[None, tf.keras.layers.Layer]:
        """
        Returns the model's output embeddings

        Returns:
            `tf.Variable`: The new weights mapping vocabulary to hidden states.
        """
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()

            try:
                return lm_head.get_output_embeddings()
            except AttributeError:
                logger.info("Building the model")
                self(self.dummy_inputs)

                return lm_head().get_output_embeddings()

        return None  # Overwrite for models with output embeddings

    def set_output_embeddings(self, value):
        """
        Set model's output embeddings

        Args:
            value (`tf.Variable`):
                The new weights mapping hidden states to vocabulary.
        """
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                lm_head.set_output_embeddings(value)
            except AttributeError:
                logger.info("Building the model")
                self(self.dummy_inputs)
                lm_head.set_output_embeddings(value)

    def get_output_layer_with_bias(self) -> Union[None, tf.keras.layers.Layer]:
        """
        Get the layer that handles a bias attribute in case the model has an LM head with weights tied to the
        embeddings

        Return:
            `tf.keras.layers.Layer`: The layer that handles the bias, None if not an LM model.
        """
        warnings.warn(
            "The method get_output_layer_with_bias is deprecated. Please use `get_lm_head` instead.", FutureWarning
        )
        return self.get_lm_head()

    def get_prefix_bias_name(self) -> Union[None, str]:
        """
        Get the concatenated _prefix name of the bias from the model name to the parent layer

        Return:
            `str`: The _prefix name of the bias.
        """
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return None

    def get_bias(self) -> Union[None, Dict[str, tf.Variable]]:
        """
        Dict of bias attached to an LM head. The key represents the name of the bias attribute.

        Return:
            `tf.Variable`: The weights representing the bias, None if not an LM model.
        """
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                return lm_head.get_bias()
            except AttributeError:
                self(self.dummy_inputs)

                return lm_head.get_bias()
        return None

    def set_bias(self, value):
        """
        Set all the bias in the LM head.

        Args:
            value (`Dict[tf.Variable]`):
                All the new bias attached to an LM head.
        """
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                lm_head.set_bias(value)
            except AttributeError:
                self(self.dummy_inputs)
                lm_head.set_bias(value)

    def get_lm_head(self) -> tf.keras.layers.Layer:
        """
        The LM Head layer. This method must be overwritten by all the models that have a lm head.

        Return:
            `tf.keras.layers.Layer`: The LM head layer if the model has one, None if not.
        """
        return None

    def resize_token_embeddings(self, new_num_tokens=None) -> tf.Variable:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `tf.Variable` module of the model without doing anything.

        Return:
            `tf.Variable`: Pointer to the input tokens Embeddings Module of the model.
        """
        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return self._get_word_embedding_weight(self.get_input_embeddings())

        model_embeds = self._resize_token_embeddings(new_num_tokens)

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens

        return model_embeds

    def _get_word_embedding_weight(model, embedding_layer):
        # If the variable holds the weights themselves, return them
        if isinstance(embedding_layer, tf.Tensor):
            return embedding_layer
        # Otherwise, try to get them from the layer's attributes

        embeds = getattr(embedding_layer, "weight", None)
        if embeds is not None:
            return embeds

        embeds = getattr(embedding_layer, "decoder", None)
        if embeds is not None:
            return embeds

        # The reason why the attributes don't exist might be
        # because the model is not built, so retry getting
        # the argument after building the model
        model(model.dummy_inputs)

        embeds = getattr(embedding_layer, "weight", None)
        if embeds is not None:
            return embeds

        embeds = getattr(embedding_layer, "decoder", None)
        if embeds is not None:
            return embeds

        return None

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self._get_word_embedding_weight(self.get_input_embeddings())
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)

        # if word embeddings are not tied, make sure that lm head bias is resized as well
        if self.get_bias() is not None:
            old_lm_head_bias = self.get_bias()
            new_lm_head_bias = self._get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)

            self.set_bias(new_lm_head_bias)

        # if word embeddings are not tied, make sure that lm head decoder is resized as well
        if self.get_output_embeddings() is not None:
            old_lm_head_decoder = self._get_word_embedding_weight(self.get_output_embeddings())
            new_lm_head_decoder = self._get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens)

            self.set_output_embeddings(new_lm_head_decoder)

        self.set_input_embeddings(new_embeddings)

        return self.get_input_embeddings()

    def _get_resized_lm_head_bias(self, old_lm_head_bias, new_num_tokens):
        """
        Build a resized bias from the old ones. Increasing the size will add newly initialized vectors at the end.
        Reducing the size will remove vectors from the end

        Args:
            old_lm_head_bias (`tf.Variable`):
                Old lm head bias to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns None

        Return:
            `tf.Variable`: Pointer to the resized bias.
        """
        new_lm_head_bias = {}

        for attr, weight in old_lm_head_bias.items():
            first_dim, old_num_tokens = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
            size_diff = new_num_tokens - old_num_tokens
            final_shape = [new_num_tokens] if first_dim is None else [first_dim, new_num_tokens]

            # initialize new bias
            if tf.math.greater(size_diff, 0):
                padding_shape = [[0, size_diff]] if first_dim is None else [[0, 0], [0, size_diff]]
                current_bias = tf.pad(weight.value(), tf.convert_to_tensor(padding_shape), constant_values=-1)
                num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
                mask_shape = [num_tokens_to_copy] if first_dim is None else [1, num_tokens_to_copy]
                bias_mask = tf.fill(tf.convert_to_tensor(mask_shape), True)
                bias_mask = tf.pad(bias_mask, tf.convert_to_tensor(padding_shape), constant_values=False)
            else:
                slice_from = [0] if first_dim is None else [0, 0]
                current_bias = tf.slice(
                    weight.value(), tf.convert_to_tensor(slice_from), tf.convert_to_tensor(final_shape)
                )
                bias_mask = tf.fill(tf.convert_to_tensor(final_shape), True)

            new_bias = self.add_weight(
                shape=final_shape,
                initializer="zeros",
                trainable=True,
                name=weight.name.split(":")[0],
            )
            init_bias = tf.where(bias_mask, current_bias, new_bias.value())

            new_bias.assign(init_bias)
            new_lm_head_bias[attr] = new_bias

        return new_lm_head_bias

    def _get_resized_lm_head_decoder(self, old_lm_head_decoder, new_num_tokens):
        """
        Build a resized decoder from the old ones. Increasing the size will add newly initialized vectors at the end.
        Reducing the size will remove vectors from the end

        Args:
            old_lm_head_decoder (`tf.Variable`):
                Old lm head decoder to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns None

        Return:
            `tf.Variable`: Pointer to the resized decoder or None if the output embeddings are different from the input
            ones.
        """
        new_lm_head_decoder = old_lm_head_decoder
        is_input_output_equals = tf.reduce_any(
            self._get_word_embedding_weight(self.get_input_embeddings()) == old_lm_head_decoder
        )

        if old_lm_head_decoder is not None and not is_input_output_equals:
            old_embedding_dim = shape_list(old_lm_head_decoder)[1]
            decoder_mask, current_decoder = init_copy_embeddings(old_lm_head_decoder, new_num_tokens)
            new_lm_head_decoder = self.add_weight(
                shape=(new_num_tokens, old_embedding_dim),
                initializer="zeros",
                trainable=True,
                name=old_lm_head_decoder.name.split(":")[0],
            )
            init_decoder = tf.where(decoder_mask, current_decoder, new_lm_head_decoder.value())

            new_lm_head_decoder.assign(init_decoder)

        return new_lm_head_decoder

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None) -> tf.Variable:
        """
        Build a resized Embedding weights from a provided token Embedding weights. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`tf.Variable`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                ``tf.Variable``` module of the model without doing anything.

        Return:
            `tf.Variable`: Pointer to the resized Embedding Module or the old Embedding Module if `new_num_tokens` is
            `None`
        """
        old_embedding_dim = shape_list(old_embeddings)[1]
        init_range = getattr(self.config, "initializer_range", 0.02)
        embeddings_mask, current_embeddings = init_copy_embeddings(old_embeddings, new_num_tokens)
        new_embeddings = self.add_weight(
            name=old_embeddings.name.split(":")[0],
            shape=[new_num_tokens, old_embedding_dim],
            initializer=get_initializer(init_range),
            dtype=tf.float32,
        )
        init_embeddings = tf.where(embeddings_mask, current_embeddings, new_embeddings.value())

        new_embeddings.assign(init_embeddings)

        return new_embeddings

    def prune_heads(self, heads_to_prune):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads
                to prune in said layer (list of `int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on
                layer 1 and heads 2 and 3 on layer 2.
        """
        raise NotImplementedError

    def save_pretrained(self, save_directory, saved_model=False, version=1, push_to_hub=False, **kwargs):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~TFPreTrainedModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str`):
                Directory to which to save. Will be created if it doesn't exist.
            saved_model (`bool`, *optional*, defaults to `False`):
                If the model has to be saved in saved model format as well or not.
            version (`int`, *optional*, defaults to 1):
                The version of the saved model. A saved model needs to be versioned in order to be properly loaded by
                TensorFlow Serving as detailed in the official documentation
                https://www.tensorflow.org/tfx/serving/serving_basic
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                <Tip warning={true}>

                Using `push_to_hub=True` will synchronize the repository you are pushing to with `save_directory`,
                which requires `save_directory` to be a local clone of the repo you are pushing to if it's an existing
                folder. Pass along `temp_dir=True` to use a temporary directory instead.

                </Tip>

            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)

        if saved_model:
            saved_model_dir = os.path.join(save_directory, "saved_model", str(version))
            self.save(saved_model_dir, include_optimizer=False, signatures=self.serving)
            logger.info(f"Saved model created in {saved_model_dir}")

        # Save configuration file
        self.config.architectures = [self.__class__.__name__[2:]]

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        self.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, TF2_WEIGHTS_NAME)
        self.save_weights(output_model_file)
        logger.info(f"Model weights saved in {output_model_file}")

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f"Model pushed to the hub in this commit: {url}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiate a pretrained TF 2.0 model from a pre-trained model configuration.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~TFPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch state_dict save file* (e.g, `./pt_model/pytorch_model.bin`). In this
                      case, `from_pt` should be set to `True` and a configuration object should be provided as `config`
                      argument. This loading path is slower than converting the PyTorch model in a TensorFlow model
                      using the provided conversion scripts and loading the TensorFlow model afterwards.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PretrainedConfig`],
                    - a string valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~TFPreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            from_pt: (`bool`, *optional*, defaults to `False`):
                Load the model weights from a PyTorch state_dict save file (see docstring of
                `pretrained_model_name_or_path` argument).
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            cache_dir (`str`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies:
                (`Dict[str, str], `optional`): A dictionary of proxy servers to use by protocol or endpoint, e.g.,
                `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
                output_loading_info(`bool`, *optional*, defaults to `False`): Whether ot not to also return a
                dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (e.g., not try doanloading the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        <Tip>

        Passing `use_auth_token=True` is required when you want to use a private model.

        </Tip>

        Examples:

        ```python
        >>> from transformers import BertConfig, TFBertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = TFBertModel.from_pretrained("bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = TFBertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = TFBertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        >>> # Loading from a Pytorch model file instead of a TensorFlow checkpoint (slower, for example purposes, not runnable).
        >>> config = BertConfig.from_json_file("./pt_model/my_pt_model_config.json")
        >>> model = TFBertModel.from_pretrained("./pt_model/my_pytorch_model.bin", from_pt=True, config=config)
        ```"""
        config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_pt = kwargs.pop("from_pt", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        load_weight_prefix = kwargs.pop("load_weight_prefix", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "model", "framework": "tensorflow", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint in priority if from_pt
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                # At this stage we don't have a weight file so we will raise an error.
                elif os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME):
                    raise EnvironmentError(
                        f"Error no file named {TF2_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} "
                        "but there is a file for PyTorch weights. Use `from_pt=True` to load this model from those "
                        "weights."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {TF2_WEIGHTS_NAME} or {WEIGHTS_NAME} found in directory "
                        f"{pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                filename = WEIGHTS_NAME if from_pt else TF2_WEIGHTS_NAME
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=filename,
                    revision=revision,
                    mirror=mirror,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                )

            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                    "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                    "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                    "login` and pass `use_auth_token=True`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                    "this model name. Check the model page at "
                    f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                if filename == TF2_WEIGHTS_NAME:
                    has_file_kwargs = {
                        "revision": revision,
                        "mirror": mirror,
                        "proxies": proxies,
                        "use_auth_token": use_auth_token,
                    }
                    if has_file(pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named {TF2_WEIGHTS_NAME} "
                            "but there is a file for PyTorch weights. Use `from_pt=True` to load this model from "
                            "those weights."
                        )
                    else:
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named {TF2_WEIGHTS_NAME} "
                            f"or {WEIGHTS_NAME}."
                        )
                else:
                    raise EnvironmentError(
                        f"{pretrained_model_name_or_path} does not appear to have a file named {filename}."
                    )
            except HTTPError as err:
                raise EnvironmentError(
                    f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n"
                    f"{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it in the cached "
                    f"files and it looks like {pretrained_model_name_or_path} is not the path to a directory "
                    f"containing a file named {TF2_WEIGHTS_NAME} or {WEIGHTS_NAME}.\n"
                    "Checkout your internet connection or see how to run the library in offline mode at "
                    "'https://huggingface.co/docs/transformers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a file named {TF2_WEIGHTS_NAME} or {WEIGHTS_NAME}."
                )

            if resolved_archive_file == archive_file:
                logger.info(f"loading weights file {archive_file}")
            else:
                logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        config.name_or_path = pretrained_model_name_or_path

        # composed models, *e.g.* TFRag, require special treatment when it comes to loading
        # pre-trained weights.
        if cls._requires_load_weight_prefix and model_kwargs.get("name") is not None:
            model_kwargs["load_weight_prefix"] = load_weight_prefix + "/" + model_kwargs.get("name")

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if from_pt:
            from .modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model

            # Load from a PyTorch checkpoint
            return load_pytorch_checkpoint_in_tf2_model(model, resolved_archive_file, allow_missing_keys=True)

        # we might need to extend the variable scope for composite models
        if load_weight_prefix is not None:
            with tf.compat.v1.variable_scope(load_weight_prefix):
                model(model.dummy_inputs)  # build the network with dummy inputs
        else:
            model(model.dummy_inputs)  # build the network with dummy inputs

        assert os.path.isfile(resolved_archive_file), f"Error retrieving file {resolved_archive_file}"
        # 'by_name' allow us to do transfer learning by skipping/adding layers
        # see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1339-L1357
        try:
            missing_keys, unexpected_keys, mismatched_keys = load_tf_weights(
                model,
                resolved_archive_file,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                _prefix=load_weight_prefix,
            )
        except OSError as e:
            try:
                with open(resolved_archive_file) as f:
                    if f.read().startswith("version"):
                        raise OSError(
                            "You seem to have cloned a repository without having git-lfs installed. Please install "
                            "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                            "you cloned."
                        )
                    else:
                        raise ValueError from e
            except (UnicodeDecodeError, ValueError):
                raise OSError(
                    "Unable to load weights from h5 file. "
                    "If you tried to load a TF 2.0 model from a PyTorch checkpoint, please set from_pt=True. "
                )

        model(model.dummy_inputs)  # Make sure restore ops are run

        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some layers from the model checkpoint at {pretrained_model_name_or_path} were not used when "
                f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
                f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n"
                f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
                f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.warning(f"All model checkpoint layers were used when initializing {model.__class__.__name__}.\n")

        if len(missing_keys) > 0:
            logger.warning(
                f"Some layers of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                f"and are newly initialized: {missing_keys}\n"
                f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.warning(
                f"All the layers of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
                f"If your task is similar to the task the model of the checkpoint was trained on, "
                f"you can already use {model.__class__.__name__} for predictions without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                f"and are newly initialized because the shapes did not match:\n{mismatched_warning}\n"
                f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
            }

            return model, loading_info

        return model


# To update the docstring, we need to copy the method, otherwise we change the original docstring.
TFPreTrainedModel.push_to_hub = copy_func(TFPreTrainedModel.push_to_hub)
TFPreTrainedModel.push_to_hub.__doc__ = TFPreTrainedModel.push_to_hub.__doc__.format(
    object="model", object_class="TFAutoModel", object_files="model checkpoint"
)


class TFConv1D(tf.keras.layers.Layer):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`):
            The number of output features.
        nx (`int`):
            The number of input features.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation to use to initialize the weights.
        kwargs:
            Additional keyword arguments passed along to the `__init__` of `tf.keras.layers.Layer`.
    """

    def __init__(self, nf, nx, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.nx = nx
        self.initializer_range = initializer_range

    def build(self, input_shape):
        self.weight = self.add_weight(
            "weight", shape=[self.nx, self.nf], initializer=get_initializer(self.initializer_range)
        )
        self.bias = self.add_weight("bias", shape=[1, self.nf], initializer=tf.zeros_initializer())

    def call(self, x):
        bz, sl = shape_list(x)[:2]

        x = tf.reshape(x, [-1, self.nx])
        x = tf.matmul(x, self.weight) + self.bias

        x = tf.reshape(x, [bz, sl, self.nf])

        return x


class TFSharedEmbeddings(tf.keras.layers.Layer):
    r"""
    Construct shared token embeddings.

    The weights of the embedding layer is usually shared with the weights of the linear decoder when doing language
    modeling.

    Args:
        vocab_size (`int`):
            The size of the vocabulary, e.g., the number of unique tokens.
        hidden_size (`int`):
            The size of the embedding vectors.
        initializer_range (`float`, *optional*):
            The standard deviation to use when initializing the weights. If no value is provided, it will default to
            \\(1/\sqrt{hidden\_size}\\).
        kwargs:
            Additional keyword arguments passed along to the `__init__` of `tf.keras.layers.Layer`.
    """

    def __init__(self, vocab_size: int, hidden_size: int, initializer_range: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = hidden_size**-0.5 if initializer_range is None else initializer_range

    def build(self, input_shape):
        """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        self.weight = self.add_weight(
            "weight", shape=[self.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range)
        )
        super().build(input_shape)

    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs: tf.Tensor, mode: str = "embedding") -> tf.Tensor:
        """
        Get token embeddings of inputs or decode final hidden state.

        Args:
            inputs (`tf.Tensor`):
                In embedding mode, should be an int64 tensor with shape `[batch_size, length]`.

                In linear mode, should be a float tensor with shape `[batch_size, length, hidden_size]`.
            mode (`str`, defaults to `"embedding"`):
               A valid value is either `"embedding"` or `"linear"`, the first one indicates that the layer should be
               used as an embedding layer, the second one that the layer should be used as a linear decoder.

        Returns:
            `tf.Tensor`: In embedding mode, the output is a float32 embedding tensor, with shape `[batch_size, length,
            embedding_size]`.

            In linear mode, the output is a float32 with shape `[batch_size, length, vocab_size]`.

        Raises:
            ValueError: if `mode` is not valid.

        Shared weights logic is adapted from
        [here](https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24).
        """
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError(f"mode {mode} is not valid.")

    def _embedding(self, input_ids):
        """Applies embedding based on inputs tensor."""
        return tf.gather(self.weight, input_ids)

    def _linear(self, inputs):
        """
        Computes logits by running inputs through a linear layer.

        Args:
            inputs: A float32 tensor with shape [..., hidden_size]

        Returns:
            float32 tensor with shape [..., vocab_size].
        """
        first_dims = shape_list(inputs)[:-1]
        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.weight, transpose_b=True)

        return tf.reshape(logits, first_dims + [self.vocab_size])


class TFSequenceSummary(tf.keras.layers.Layer):
    """
    Compute a single vector summary of a sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_type** (`str`) -- The method to use to make this summary. Accepted values are:

                - `"last"` -- Take the last token hidden state (like XLNet)
                - `"first"` -- Take the first token hidden state (like Bert)
                - `"mean"` -- Take the mean of all tokens hidden states
                - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
                - `"attn"` -- Not implemented now, use multi-head attention

            - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
              (otherwise to `config.hidden_size`).
            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string or `None` will add no activation.
            - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
            - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.

        initializer_range (`float`, defaults to 0.02): The standard deviation to use to initialize the weights.
        kwargs:
            Additional keyword arguments passed along to the `__init__` of `tf.keras.layers.Layer`.
    """

    def __init__(self, config: PretrainedConfig, initializer_range: float = 0.02, **kwargs):
        super().__init__(**kwargs)

        self.summary_type = config.summary_type if hasattr(config, "summary_use_proj") else "last"
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.has_summary = hasattr(config, "summary_use_proj") and config.summary_use_proj
        if self.has_summary:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = tf.keras.layers.Dense(
                num_classes, kernel_initializer=get_initializer(initializer_range), name="summary"
            )

        self.has_activation = False
        activation_string = getattr(config, "summary_activation", None)
        if activation_string is not None:
            self.has_activation = True
            self.activation = get_tf_activation(activation_string)

        self.has_first_dropout = hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0
        if self.has_first_dropout:
            self.first_dropout = tf.keras.layers.Dropout(config.summary_first_dropout)

        self.has_last_dropout = hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0
        if self.has_last_dropout:
            self.last_dropout = tf.keras.layers.Dropout(config.summary_last_dropout)

    def call(self, inputs, cls_index=None, training=False):
        if not isinstance(inputs, (dict, tuple, list)):
            hidden_states = inputs
        elif isinstance(inputs, (tuple, list)):
            hidden_states = inputs[0]
            cls_index = inputs[1] if len(inputs) > 1 else None
            assert len(inputs) <= 2, "Too many inputs."
        else:
            hidden_states = inputs.get("hidden_states")
            cls_index = inputs.get("cls_index", None)

        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = tf.reduce_mean(hidden_states, axis=1)
        elif self.summary_type == "cls_index":
            hidden_shape = shape_list(hidden_states)  # e.g. [batch, num choices, seq length, hidden dims]
            if cls_index is None:
                cls_index = tf.fill(
                    hidden_shape[:-2], hidden_shape[-2] - 1
                )  # A tensor full of shape [batch] or [batch, num choices] full of sequence length
            cls_shape = shape_list(cls_index)
            if len(cls_shape) <= len(hidden_shape) - 2:
                cls_index = tf.expand_dims(cls_index, axis=-1)
            # else:
            # cls_index = cls_index[..., tf.newaxis]
            # cls_index = cls_index.expand((-1,) * (cls_index.dim()-1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = tf.gather(hidden_states, cls_index, batch_dims=len(hidden_shape) - 2)
            output = tf.squeeze(
                output, axis=len(hidden_shape) - 2
            )  # shape of output: (batch, num choices, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        if self.has_first_dropout:
            output = self.first_dropout(output, training=training)

        if self.has_summary:
            output = self.summary(output)

        if self.has_activation:
            output = self.activation(output)

        if self.has_last_dropout:
            output = self.last_dropout(output, training=training)

        return output

    @classmethod
    def register_for_auto_class(cls, auto_class="TFAutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"TFAutoModel"`):
                The auto class to register this new model with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class


def get_initializer(initializer_range: float = 0.02) -> tf.initializers.TruncatedNormal:
    """
    Creates a `tf.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (*float*, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        `tf.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


class TFWrappedEmbeddings:
    """
    this class wraps a the TFSharedEmbeddingTokens layer into a python 'no-keras-layer' class to avoid problem with
    weight restoring. Also it makes sure that the layer is called from the correct scope to avoid problem with
    saving/storing the correct weights
    """

    def __init__(self, layer, abs_scope_name=None):
        self._layer = layer
        self._abs_scope_name = abs_scope_name

    def call(self, inputs, mode="embedding"):
        if self._abs_scope_name is None:
            return self._layer.call(inputs, mode)

        # if an abs scope name is given to the embedding variable, call variable from absolute scope
        with tf.compat.v1.variable_scope(self._abs_scope_name, auxiliary_name_scope=False) as abs_scope_name:
            with tf.name_scope(abs_scope_name.original_name_scope):
                return self._layer.call(inputs, mode)

    def __call__(self, inputs, mode="embedding"):
        if self._abs_scope_name is None:
            return self._layer(inputs, mode)

        # if an abs scope name is given to the embedding variable, call variable from absolute scope
        with tf.compat.v1.variable_scope(self._abs_scope_name, auxiliary_name_scope=False) as abs_scope_name:
            with tf.name_scope(abs_scope_name.original_name_scope):
                return self._layer(inputs, mode)
