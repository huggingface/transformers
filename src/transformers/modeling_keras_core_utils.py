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

from __future__ import annotations

import inspect
import json
import os
import pickle
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import h5py
import tensorflow as tf
from huggingface_hub import Repository, list_repo_files
import keras_core as keras
import keras_core.backend as K
from packaging.version import parse

from . import DataCollatorWithPadding, DefaultDataCollator
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, TFGenerationMixin
from .tf_utils import (
    expand_1d,
    save_attributes_to_hdf5_group,
    shape_list,
)
from .utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_INDEX_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ModelOutput,
    PushToHubMixin,
    cached_file,
    download_url,
    find_labels,
    has_file,
    is_offline_mode,
    is_remote_url,
    is_safetensors_available,
    logging,
    requires_backends,
    working_or_temp_dir,
)
from .utils.hub import get_checkpoint_shard_files
from .modeling_tf_utils import dummy_loss, TFModelUtilsMixin, init_copy_embeddings, tf_shard_checkpoint, format_weight_name, get_initializer, tf_logger


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.tensorflow import save_file as safe_save_file

if TYPE_CHECKING:
    from . import PreTrainedTokenizerBase


logger = logging.get_logger(__name__)

class KerasPreTrainedModel(keras.Model, TFModelUtilsMixin, TFGenerationMixin, PushToHubMixin):
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
    _label_to_output_map = None

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
        dummies = {}
        for key, spec in self.input_signature.items():
            # 2 is the most correct arbitrary size. I will not be taking questions
            dummy_shape = [dim if dim is not None else 2 for dim in spec.shape]
            if spec.shape[0] is None:
                # But let's make the batch size 1 to save memory anyway
                dummy_shape[0] = 1
            dummies[key] = tf.ones(shape=dummy_shape, dtype=spec.dtype)
            if key == "token_type_ids":
                # Some models have token_type_ids but with a vocab_size of 1
                dummies[key] = tf.zeros_like(dummies[key])
        if self.config.add_cross_attention and "encoder_hidden_states" in inspect.signature(self.call).parameters:
            if "encoder_hidden_states" not in dummies:
                if self.main_input_name == "input_ids":
                    dummies["encoder_hidden_states"] = tf.ones(
                        shape=(1, 2, self.config.hidden_size), dtype=tf.float32, name="encoder_hidden_states"
                    )
                else:
                    raise NotImplementedError(
                        "Model has cross-attention but we couldn't infer the shape for the encoder hidden states. Please manually override dummy_inputs!"
                    )
        return dummies

    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a TensorFlow model.
        """
        return "tf"

    def build(self, input_shape=None):
        # TODO Matt: Temporary workaround since we don't have the call-context function, we just skip that check
        if not self.built:
            self.built = True
            # Set the serving spec quickly to ensure that Keras doesn't use the specific dummy input shapes as the spec
            # Setting it in build() allows users to override the shape when loading a non-pretrained model from config
            # TODO Matt: self._set_save_spec() was used here, and might need to be replaced, since it doesn't exist in Keras-Core.
            self(self.dummy_inputs, training=False)

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
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

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

    def get_head_mask(self, head_mask: tf.Tensor | None, num_hidden_layers: int) -> tf.Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`tf.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.

        Returns:
            `tf.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.shape.rank == 1:
            head_mask = head_mask[None, None, :, None, None]
            head_mask = tf.repeat(head_mask, repeats=num_hidden_layers, axis=0)
        elif head_mask.shape.rank == 2:
            head_mask = head_mask[:, None, :, None, None]
        assert head_mask.shape.rank == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = tf.cast(head_mask, tf.float32)  # switch to float if need + fp16 compatibility
        return head_mask

    @tf.function
    def serving(self, inputs):
        """
        Args:
        Method used for serving the model. Does not have a specific signature, but will be specialized as concrete
        functions when saving with `save_pretrained`.
            inputs (`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """
        output = self.call(inputs)

        return self.serving_output(output)

    def eager_serving(self, inputs):
        """
        Method used for serving the model. This method is deprecated, and will be removed.

        Args:
            inputs (`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """
        warnings.warn(
            "The function `eager_serving` is deprecated and will be removed in version 4.32.0 of Transformers",
            FutureWarning,
        )
        output = self.call(inputs)

        return self.serving_output(output)

    @property
    def input_signature(self) -> Dict[str, tf.TensorSpec]:
        """
        This property should return a dict mapping input names to tf.TensorSpec objects, representing the expected
        shape and dtype for model inputs. It is used for both serving and for generating the dummy inputs used to build
        the model.
        """
        model_inputs = list(inspect.signature(self.call).parameters)
        sig = {}
        if "input_ids" in model_inputs:
            if self.__class__.__name__.endswith("ForMultipleChoice"):
                text_dims = 3
            else:
                text_dims = 2
            for input_name in (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "decoder_input_ids",
                "decoder_attention_mask",
            ):
                if input_name in model_inputs:
                    sig[input_name] = tf.TensorSpec([None] * text_dims, tf.int32, name=input_name)
        if "pixel_values" in model_inputs:
            pixel_values_shape = [None, None, None, None]
            if hasattr(self.config, "vision_config"):
                vision_config = self.config.vision_config
            else:
                vision_config = self.config
            if hasattr(vision_config, "num_channels"):
                pixel_values_shape[1] = vision_config.num_channels
            else:
                raise NotImplementedError(
                    "Could not infer number of channels from config, please override input_signature to specify input shapes."
                )
            if hasattr(vision_config, "image_size"):
                pixel_values_shape[2] = pixel_values_shape[3] = vision_config.image_size
            elif hasattr(vision_config, "input_size"):
                pixel_values_shape[2] = pixel_values_shape[3] = vision_config.input_size
            else:
                raise NotImplementedError(
                    "Could not infer input image shape from config, please override input_signature to specify input shapes."
                )
            sig["pixel_values"] = tf.TensorSpec(pixel_values_shape, tf.float32, name="pixel_values")
        if "input_features" in model_inputs:
            raise NotImplementedError("Audio models need a manually defined input_signature")
        return sig

    def serving_output(self, output):
        """
        Prepare the output of the saved model. Can be overridden if specific serving modifications are required.
        """
        if not isinstance(output, ModelOutput):
            return output
        for key in output:
            if key.endswith("hidden_states") and not getattr(self.config, "output_hidden_states", False):
                output[key] = None
            elif key.endswith("attentions") and not getattr(self.config, "output_attentions", False):
                output[key] = None
            elif key == "past_key_values" and not getattr(self.config, "use_cache", False):
                output[key] = None
            elif key == "cross_attentions" and not (
                getattr(self.config, "output_attentions", False) and getattr(self.config, "add_cross_attention", False)
            ):
                output[key] = None
            if isinstance(output[key], (tuple, list)):
                try:
                    output[key] = tf.convert_to_tensor(output[key])
                except (ValueError, tf.errors.InvalidArgumentError):
                    pass  # Layers may not have the same dimensions
        return output

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation.
        # Alternativelly, the model can also have a custom `generate` function.
        if "GenerationMixin" in str(cls.prepare_inputs_for_generation) and "GenerationMixin" in str(cls.generate):
            return False
        return True

    def get_input_embeddings(self) -> keras.Layer:
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
        if os.path.isdir(repo_path_or_name):
            local_dir = repo_path_or_name
        else:
            # If this isn't a local path, check that the remote repo exists and has a checkpoint in it
            repo_files = list_repo_files(repo_path_or_name)
            for file in ("checkpoint/weights.h5", "checkpoint/extra_data.pickle"):
                if file not in repo_files:
                    raise FileNotFoundError(f"Repo {repo_path_or_name} does not contain checkpoint file {file}!")
            repo = Repository(repo_path_or_name.split("/")[-1], clone_from=repo_path_or_name)
            local_dir = repo.local_dir

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

    def prepare_tf_dataset(
        self,
        dataset: "datasets.Dataset",  # noqa:F821
        batch_size: int = 8,
        shuffle: bool = True,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        collate_fn: Optional[Callable] = None,
        collate_fn_args: Optional[Dict[str, Any]] = None,
        drop_remainder: Optional[bool] = None,
        prefetch: bool = True,
    ):
        """
        Wraps a HuggingFace [`~datasets.Dataset`] as a `tf.data.Dataset` with collation and batching. This method is
        designed to create a "ready-to-use" dataset that can be passed directly to Keras methods like `fit()` without
        further modification. The method will drop columns from the dataset if they don't match input names for the
        model. If you want to specify the column names to return rather than using the names that match this model, we
        recommend using `Dataset.to_tf_dataset()` instead.

        Args:
            dataset (`Any`):
                A [~`datasets.Dataset`] to be wrapped as a `tf.data.Dataset`.
            batch_size (`int`, defaults to 8):
                The size of batches to return.
            shuffle (`bool`, defaults to `True`):
                Whether to return samples from the dataset in random order. Usually `True` for training datasets and
                `False` for validation/test datasets.
            tokenizer ([`PreTrainedTokenizerBase`], *optional*):
                A `PreTrainedTokenizer` that will be used to pad samples to create batches. Has no effect if a specific
                `collate_fn` is passed instead.
            collate_fn (`Callable`, *optional*):
                A function that collates samples from the dataset into a single batch. Defaults to
                `DefaultDataCollator` if no `tokenizer` is supplied or `DataCollatorWithPadding` if a `tokenizer` is
                passed.
            collate_fn_args (`Dict[str, Any]`, *optional*):
                A dict of arguments to pass to the `collate_fn` alongside the list of samples.
            drop_remainder (`bool`, *optional*):
                Whether to drop the final batch, if the batch_size does not evenly divide the dataset length. Defaults
                to the same setting as `shuffle`.
            prefetch (`bool`, defaults to `True`):
                Whether to add prefetching to the end of the `tf.data` pipeline. This is almost always beneficial for
                performance, but can be disabled in edge cases.


        Returns:
            `Dataset`: A `tf.data.Dataset` which is ready to pass to the Keras API.
        """
        requires_backends(self, ["datasets"])
        import datasets

        if collate_fn is None:
            if tokenizer is None:
                collate_fn = DefaultDataCollator(return_tensors="np")
            else:
                collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="np")
        if collate_fn_args is None:
            collate_fn_args = {}

        if not isinstance(dataset, datasets.Dataset):
            raise TypeError("Dataset argument should be a datasets.Dataset!")
        model_inputs = list(inspect.signature(self.call).parameters)
        model_labels = find_labels(self.__class__)
        if "cols_to_retain" in list(inspect.signature(dataset._get_output_signature).parameters.keys()):
            output_signature, _ = dataset._get_output_signature(
                dataset,
                batch_size=None,
                collate_fn=collate_fn,
                collate_fn_args=collate_fn_args,
                cols_to_retain=model_inputs,
            )
        else:
            # TODO Matt: This is a workaround for older versions of datasets that are missing the `cols_to_retain`
            #            argument. We should remove this once the minimum supported version of datasets is > 2.3.2
            unwanted_columns = [
                feature
                for feature in dataset.features
                if feature not in model_inputs and feature not in ("label_ids", "label")
            ]
            dataset = dataset.remove_columns(unwanted_columns)
            output_signature, _ = dataset._get_output_signature(
                dataset, batch_size=None, collate_fn=collate_fn, collate_fn_args=collate_fn_args
            )
        output_columns = list(output_signature.keys())
        feature_cols = [col for col in output_columns if col in model_inputs and col not in model_labels]
        label_cols = [col for col in output_columns if col in model_labels]

        # Backwards compatibility for older versions of datasets. Previously, if `columns` or `label_cols`
        # were a single element list, the returned element spec would be a single element. Now, passing [feature]
        # will return a dict structure {"feature": feature}, and passing a single string will return a single element.
        feature_cols = feature_cols[0] if len(feature_cols) == 1 else feature_cols
        label_cols = label_cols[0] if len(label_cols) == 1 else label_cols

        if drop_remainder is None:
            drop_remainder = shuffle
        tf_dataset = dataset.to_tf_dataset(
            columns=feature_cols,
            label_cols=label_cols,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_remainder=drop_remainder,
            collate_fn=collate_fn,
            collate_fn_args=collate_fn_args,
            prefetch=prefetch,
        )
        return tf_dataset

    def compile(
        self,
        optimizer="rmsprop",
        loss="auto_with_warning",
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs,
    ):
        """
        This is a thin wrapper that sets the model's loss output head as the loss if the user does not specify a loss
        function themselves.
        """
        if loss in ("auto_with_warning", "passthrough"):  # "passthrough" for workflow backward compatibility
            logger.info(
                "No loss specified in compile() - the model's internal loss computation will be used as the "
                "loss. Don't panic - this is a common way to train TensorFlow models in Transformers! "
                "To disable this behaviour please pass a loss argument, or explicitly pass "
                "`loss=None` if you do not want your model to compute a loss. You can also specify `loss='auto'` to "
                "get the internal loss without printing this info string."
            )
            loss = "auto"
        if loss == "auto":
            loss = dummy_loss
            self._using_dummy_loss = True
        else:
            self._using_dummy_loss = False
        parent_args = list(inspect.signature(keras.Model.compile).parameters.keys())
        # This argument got renamed, we need to support both versions
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
        if hasattr(keras.Model, "compute_loss"):
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

    def get_label_to_output_name_mapping(self):
        arg_names = list(inspect.signature(self.call).parameters)
        if self._label_to_output_map is not None:
            return self._label_to_output_map
        elif "start_positions" in arg_names:
            return {"start_positions": "start_logits", "end_positions": "end_logits"}
        elif "sentence_order_label" in arg_names:
            return {"labels": "prediction_logits", "sentence_order_label": "sop_logits"}
        elif "next_sentence_label" in arg_names:
            return {"labels": "prediction_logits", "next_sentence_label": "seq_relationship_logits"}
        elif "mc_labels" in arg_names:
            return {"labels": "logits", "mc_labels": "mc_logits"}
        else:
            return {}

    def train_step(self, data):
        """
        A modification of Keras's default `train_step` that correctly handles matching outputs to labels for our models
        and supports directly training on the loss output head. In addition, it ensures input keys are copied to the
        labels where appropriate. It will also copy label keys into the input dict when using the dummy loss, to ensure
        that they are available to the model during the forward pass.
        """

        # We hardcode the most common renamings; models with weirder names can set `self._label_to_output_map`
        arg_names = list(inspect.signature(self.call).parameters)
        label_kwargs = find_labels(self.__class__)
        label_to_output = self.get_label_to_output_name_mapping()
        output_to_label = {val: key for key, val in label_to_output.items()}
        if not self._using_dummy_loss and parse(tf.__version__) < parse("2.11.0"):
            # Newer TF train steps leave this out
            data = expand_1d(data)
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # If the inputs are mutable dictionaries, make a shallow copy of them because we will modify
        # them during input/label pre-processing. This avoids surprising the user by wrecking their data.
        # In addition, modifying mutable Python inputs makes XLA compilation impossible.
        if isinstance(x, dict):
            x = x.copy()
        if isinstance(y, dict):
            y = y.copy()

        # When using a dummy loss, we ensure that separate labels are copied to the correct model arguments,
        # if those keys are not already present in the input dict
        if self._using_dummy_loss and y is not None:
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
                    elif output_to_label.get(key, None) in arg_names and key not in x:
                        x[output_to_label[key]] = val
        if y is None:
            y = {key: val for key, val in x.items() if key in label_kwargs}
            if not y and not self._using_dummy_loss:
                raise ValueError("Could not find label column(s) in input dict and no separate labels were provided!")

        if isinstance(y, dict):
            # Rename labels at this point to match output heads
            y = {label_to_output.get(key, key): val for key, val in y.items()}

        # Run forward pass.
        with tf.GradientTape() as tape:
            if self._using_dummy_loss and "return_loss" in arg_names:
                y_pred = self(x, training=True, return_loss=True)
            else:
                y_pred = self(x, training=True)
            if self._using_dummy_loss:
                loss = self.compiled_loss(y_pred.loss, y_pred.loss, sample_weight, regularization_losses=self.losses)
            else:
                loss = None

            # This next block matches outputs to label keys. Tensorflow's standard method for doing this
            # can get very confused if any of the keys contain nested values (e.g. lists/tuples of Tensors)
            if isinstance(y, dict) and len(y) == 1:
                if list(y.keys())[0] in y_pred.keys():
                    y_pred = y_pred[list(y.keys())[0]]
                elif list(y_pred.keys())[0] == "loss":
                    y_pred = y_pred[1]
                else:
                    y_pred = y_pred[0]
                _, y = y.popitem()
            elif isinstance(y, dict):
                # If the labels are a dict, match keys from the output by name
                y_pred = {key: val for key, val in y_pred.items() if key in y}
            elif isinstance(y, tuple) or isinstance(y, list):
                # If the labels are a tuple/list, match keys to the output by order, skipping the loss.
                if list(y_pred.keys())[0] == "loss":
                    y_pred = y_pred.to_tuple()[1:]
                else:
                    y_pred = y_pred.to_tuple()
                y_pred = y_pred[: len(y)]  # Remove unused fields in case those cause problems
            else:
                # If the labels are a single tensor, match them to the first non-loss tensor in the output
                if list(y_pred.keys())[0] == "loss":
                    y_pred = y_pred[1]
                else:
                    y_pred = y_pred[0]

            if loss is None:
                loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def test_step(self, data):
        """
        A modification of Keras's default `train_step` that correctly handles matching outputs to labels for our models
        and supports directly training on the loss output head. In addition, it ensures input keys are copied to the
        labels where appropriate. It will also copy label keys into the input dict when using the dummy loss, to ensure
        that they are available to the model during the forward pass.
        """
        # We hardcode the most common renamings; models with weirder names can set `self._label_to_output_map`
        arg_names = list(inspect.signature(self.call).parameters)
        label_kwargs = find_labels(self.__class__)
        label_to_output = self.get_label_to_output_name_mapping()
        output_to_label = {val: key for key, val in label_to_output.items()}
        if not self._using_dummy_loss and parse(tf.__version__) < parse("2.11.0"):
            # Newer versions leave this out
            data = expand_1d(data)
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # If the inputs are mutable dictionaries, make a shallow copy of them because we will modify
        # them during input/label pre-processing. This avoids surprising the user by wrecking their data.
        # In addition, modifying mutable Python inputs makes XLA compilation impossible.
        if isinstance(x, dict):
            x = x.copy()
        if isinstance(y, dict):
            y = y.copy()

        # When using a dummy loss, we ensure that separate labels are copied to the correct model arguments,
        # if those keys are not already present in the input dict
        if self._using_dummy_loss and y is not None:
            arg_names = list(inspect.signature(self.call).parameters)
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
                    elif output_to_label.get(key, None) in arg_names and key not in x:
                        x[output_to_label[key]] = val
        if y is None:
            y = {key: val for key, val in x.items() if key in label_kwargs}
            if not y and not self._using_dummy_loss:
                raise ValueError("Could not find label column(s) in input dict and no separate labels were provided!")

        if isinstance(y, dict):
            # Rename labels at this point to match output heads
            y = {label_to_output.get(key, key): val for key, val in y.items()}

        # Run forward pass.
        if self._using_dummy_loss and "return_loss" in arg_names:
            y_pred = self(x, return_loss=True, training=False)
        else:
            y_pred = self(x, training=False)
        if self._using_dummy_loss:
            loss = self.compiled_loss(y_pred.loss, y_pred.loss, sample_weight, regularization_losses=self.losses)
        else:
            loss = None

        # This next block matches outputs to label keys. Tensorflow's standard method for doing this
        # can get very confused if any of the keys contain nested values (e.g. lists/tuples of Tensors)
        if isinstance(y, dict) and len(y) == 1:
            if list(y.keys())[0] in y_pred.keys():
                y_pred = y_pred[list(y.keys())[0]]
            elif list(y_pred.keys())[0] == "loss":
                y_pred = y_pred[1]
            else:
                y_pred = y_pred[0]
            _, y = y.popitem()
        elif isinstance(y, dict):
            # If the labels are a dict, match keys from the output by name
            y_pred = {key: val for key, val in y_pred.items() if key in y}
        elif isinstance(y, tuple) or isinstance(y, list):
            # If the labels are a tuple/list, match keys to the output by order, skipping the loss.
            if list(y_pred.keys())[0] == "loss":
                y_pred = y_pred.to_tuple()[1:]
            else:
                y_pred = y_pred.to_tuple()
            y_pred = y_pred[: len(y)]  # Remove unused fields in case those cause problems
        else:
            # If the labels are a single tensor, match them to the first non-loss tensor in the output
            if list(y_pred.keys())[0] == "loss":
                y_pred = y_pred[1]
            else:
                y_pred = y_pred[0]

        if loss is None:
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
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
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            output_dir (`str` or `os.PathLike`):
                The folder in which to create the model card.
            model_name (`str`, *optional*):
                The name of the model.
            language (`str`, *optional*):
                The language of the model (if applicable)
            license (`str`, *optional*):
                The license of the model. Will default to the license of the pretrained model used, if the original
                model given to the `Trainer` comes from a repo on the Hub.
            tags (`str` or `List[str]`, *optional*):
                Some tags to be included in the metadata of the model card.
            finetuned_from (`str`, *optional*):
                The name of the model used to fine-tune this one (if applicable). Will default to the name of the repo
                of the original model given to the `Trainer` (if it comes from the Hub).
            tasks (`str` or `List[str]`, *optional*):
                One or several task identifiers, to be included in the metadata of the model card.
            dataset_tags (`str` or `List[str]`, *optional*):
                One or several dataset tags, to be included in the metadata of the model card.
            dataset (`str` or `List[str]`, *optional*):
                One or several dataset identifiers, to be included in the metadata of the model card.
            dataset_args (`str` or `List[str]`, *optional*):
               One or several dataset arguments, to be included in the metadata of the model card.
        """
        # Avoids a circular import by doing this when necessary.
        from .modelcard import TrainingSummary  # tests_ignore

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
            self.build()
            main_layer.set_input_embeddings(value)

    def get_output_embeddings(self) -> Union[None, keras.Layer]:
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
                self.build()

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
                self.build()
                lm_head.set_output_embeddings(value)

    def get_output_layer_with_bias(self) -> Union[None, keras.Layer]:
        """
        Get the layer that handles a bias attribute in case the model has an LM head with weights tied to the
        embeddings

        Return:
            `keras.Layer`: The layer that handles the bias, None if not an LM model.
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
                self.build()

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
                self.build()
                lm_head.set_bias(value)

    def get_lm_head(self) -> keras.Layer:
        """
        The LM Head layer. This method must be overwritten by all the models that have a lm head.

        Return:
            `keras.Layer`: The LM head layer if the model has one, None if not.
        """
        return None

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None
    ) -> Union[tf.keras.layers.Embedding, tf.Variable]:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens without doing anything.

        Return:
            `tf.Variable` or `tf.keras.layers.Embedding`: Pointer to the input tokens of the model.
        """
        # TODO (joao): flagged for replacement (by `_v2_resized_token_embeddings`) due to embeddings refactor

        # Run the new code path if the model has a keras embeddings layer
        if isinstance(self.get_input_embeddings(), tf.keras.layers.Embedding):
            return self._v2_resized_token_embeddings(new_num_tokens)

        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return self._get_word_embedding_weight(self.get_input_embeddings())

        model_embeds = self._resize_token_embeddings(new_num_tokens)

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens

        return model_embeds

    def _v2_resized_token_embeddings(self, new_num_tokens: Optional[int] = None) -> tf.keras.layers.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens without doing anything.

        Return:
            `tf.keras.layers.Embedding`: Pointer to the input tokens of the model.
        """
        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return self.get_input_embeddings()

        model_embeds = self._v2_resize_token_embeddings(new_num_tokens)

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens

        return model_embeds

    def _get_word_embedding_weight(model, embedding_layer):
        # TODO (joao): flagged for delection due to embeddings refactor

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
        model.build()

        embeds = getattr(embedding_layer, "weight", None)
        if embeds is not None:
            return embeds

        embeds = getattr(embedding_layer, "decoder", None)
        if embeds is not None:
            return embeds

        return None

    def _resize_token_embeddings(self, new_num_tokens):
        # TODO (joao): flagged for replacement (by `_v2_resize_token_embeddings`) due to embeddings refactor
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

    def _v2_resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._v2_get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # If word embeddings are not tied, make sure that lm head bias is resized as well
        if self.get_bias() is not None:
            old_lm_head_bias = self.get_bias()
            new_lm_head_bias = self._v2_get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)
            self.set_bias(new_lm_head_bias)

        # If word embeddings are not tied, make sure that lm head decoder is resized as well.
        tied_weights = self.get_input_embeddings() == self.get_output_embeddings()
        if self.get_output_embeddings() is not None and not tied_weights:
            old_lm_head_decoder = self._get_word_embedding_weight(self.get_output_embeddings())
            # TODO (joao): this one probably needs a v2 version with other models
            new_lm_head_decoder = self._get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens)
            self.set_output_embeddings(new_lm_head_decoder)

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
        # TODO (joao): flagged for replacement (by `_v2_get_resized_lm_head_bias`) due to embeddings refactor
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

    def _v2_get_resized_lm_head_bias(
        self, old_lm_head_bias: Dict[str, tf.Variable], new_num_tokens: int
    ) -> Dict[str, tf.Tensor]:
        """
        Build a resized bias from the old ones. Increasing the size will add newly initialized vectors at the end.
        Reducing the size will remove vectors from the end

        Args:
            old_lm_head_bias (`Dict[str, tf.Variable]`):
                Old lm head bias to be resized.
            new_num_tokens (`int`):
                New number of tokens in the linear matrix. Increasing the size will add newly initialized vectors at
                the end. Reducing the size will remove vectors from the end.

        Return:
            `tf.Tensor`: Values for the resized bias.
        """
        new_lm_head_bias = {}

        for attr, weight in old_lm_head_bias.items():
            # Determine the size difference (depending on the shape)
            first_dim, old_num_tokens = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
            size_diff = new_num_tokens - old_num_tokens

            # Copy the old bias values to the new bias
            if old_num_tokens > new_num_tokens:
                new_bias = weight.value()[..., :new_num_tokens]
            else:
                padding_shape = [[0, size_diff]] if first_dim is None else [[0, 0], [0, size_diff]]
                new_bias = tf.pad(weight.value(), tf.convert_to_tensor(padding_shape))

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
                `tf.Variable` module of the model without doing anything.

        Return:
            `tf.Variable`: Pointer to the resized Embedding Module or the old Embedding Module if `new_num_tokens` is
            `None`
        """
        # TODO (joao): flagged for replacement (by `_v2_get_resized_embeddings`) due to embeddings refactor
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

    def _v2_get_resized_embeddings(
        self, old_embeddings: tf.keras.layers.Embedding, new_num_tokens: int
    ) -> tf.keras.layers.Embedding:
        """
        Build a resized Embedding layer from a provided Embedding layer. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end.

        Args:
            old_embeddings (`tf.keras.layers.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

        Return:
            `tf.keras.layers.Embedding`: Resized Embedding layer.
        """

        # Get the initialization range for the embeddings
        init_range = 0.02  # default value
        potential_initialization_variable_names = [
            "initializer_range",  # most common
            "initializer_factor",  # e.g. T5
            "init_std",  # e.g BART
        ]
        for var_name in potential_initialization_variable_names:
            if hasattr(self.config, var_name):
                init_range = getattr(self.config, var_name)

        # Get a new (initialized) embeddings layer
        new_embeddings = tf.keras.layers.Embedding(
            input_dim=new_num_tokens,
            output_dim=old_embeddings.output_dim,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=init_range),
            name=old_embeddings.embeddings.name[:-13],  # exact same scoped name except "/embeddings:0"
        )
        new_embeddings(tf.constant([[0]]))

        # Copy the old embeddings to the new embeddings
        if old_embeddings.input_dim >= new_num_tokens:
            init_embeddings = old_embeddings.embeddings[:new_num_tokens]
        else:
            init_embeddings = tf.concat(
                [old_embeddings.embeddings, new_embeddings.embeddings[old_embeddings.input_dim :]], axis=0
            )
        new_embeddings.embeddings.assign(init_embeddings)
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

    def save_pretrained(
        self,
        save_directory,
        saved_model=False,
        version=1,
        push_to_hub=False,
        signatures=None,
        max_shard_size: Union[int, str] = "10GB",
        create_pr: bool = False,
        safe_serialization: bool = False,
        token: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
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
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            signatures (`dict` or `tf.function`, *optional*):
                Model's signature used for serving. This will be passed to the `signatures` argument of model.save().
            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            safe_serialization (`bool`, *optional*, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        if saved_model:
            # If `torch_dtype` is in the config with a torch dtype class as the value, we need to change it to string.
            # (Although TF doesn't care about this attribute, we can't just remove it or set it to `None`.)
            if getattr(self.config, "torch_dtype", None) is not None and not isinstance(self.config.torch_dtype, str):
                self.config.torch_dtype = str(self.config.torch_dtype).split(".")[1]
            if signatures is None:
                serving_default = self.serving.get_concrete_function(self.input_signature)
                if any(spec.dtype == tf.int32 for spec in self.input_signature.values()):
                    int64_spec = {
                        key: tf.TensorSpec(
                            shape=spec.shape, dtype=tf.int64 if spec.dtype == tf.int32 else spec.dtype, name=spec.name
                        )
                        for key, spec in self.input_signature.items()
                    }
                    int64_serving = self.serving.get_concrete_function(int64_spec)
                    signatures = {"serving_default": serving_default, "int64_serving": int64_serving}
                else:
                    signatures = serving_default
            saved_model_dir = os.path.join(save_directory, "saved_model", str(version))
            self.save(saved_model_dir, include_optimizer=False, signatures=signatures)
            logger.info(f"Saved model created in {saved_model_dir}")

        # Save configuration file
        self.config.architectures = [self.__class__.__name__[2:]]

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        self.config.save_pretrained(save_directory)
        if self.can_generate():
            self.generation_config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else TF2_WEIGHTS_NAME
        output_model_file = os.path.join(save_directory, weights_name)

        shards, index = tf_shard_checkpoint(self.weights, max_shard_size)

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")
            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
            ):
                os.remove(full_filename)

        if index is None:
            if safe_serialization:
                state_dict = {format_weight_name(w.name): w.value() for w in self.weights}
                safe_save_file(state_dict, output_model_file, metadata={"format": "tf"})
            else:
                self.save_weights(output_model_file)
            logger.info(f"Model weights saved in {output_model_file}")
        else:
            save_index_file = os.path.join(save_directory, TF2_WEIGHTS_INDEX_NAME)
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as index_file:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                index_file.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )
            for shard_file, shard in shards.items():
                with h5py.File(os.path.join(save_directory, shard_file), mode="w") as shard_file:
                    layers = []
                    for layer in sorted(shard, key=lambda x: x.name):
                        if "model." in layer.name or len(layer.name.split("/")) == 1:
                            layer_name = layer.name
                        else:
                            layer_name = "/".join(layer.name.split("/")[1:])
                        param_dset = shard_file.create_dataset(
                            layer_name, layer.numpy().shape, dtype=layer.numpy().dtype
                        )
                        param_dset[:] = layer.numpy()
                        layers.append(layer_name.encode("utf8"))
                    save_attributes_to_hdf5_group(shard_file, "layer_names", layers)

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
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
            from_pt (`bool`, *optional*, defaults to `False`):
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
                Whether or not to only look at local files (e.g., not try downloading the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.


                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            tf_to_pt_weight_rename (`Callable`, *optional*):
                A function that is called to transform the names of weights during the PyTorch to TensorFlow
                crossloading process. This is not necessary for most models, but is useful to allow composite models to
                be crossloaded correctly.
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
        from_pt = kwargs.pop("from_pt", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        load_weight_prefix = kwargs.pop("load_weight_prefix", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        tf_to_pt_weight_rename = kwargs.pop("tf_to_pt_weight_rename", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

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
                token=token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                _commit_hash=commit_hash,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        if commit_hash is None:
            commit_hash = getattr(config, "_commit_hash", None)

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint in priority if from_pt
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                elif from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)
                    is_sharded = True
                elif is_safetensors_available() and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
                ):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
                elif is_safetensors_available() and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
                    is_sharded = True
                    raise NotImplementedError("Support for sharded checkpoints using safetensors is coming soon!")
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_INDEX_NAME)):
                    # Load from a sharded TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_INDEX_NAME)
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)) or os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)
                ):
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
            elif os.path.isfile(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                archive_file = pretrained_model_name_or_path + ".index"
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                # set correct filename
                if from_pt:
                    filename = WEIGHTS_NAME
                elif is_safetensors_available():
                    filename = SAFE_WEIGHTS_NAME
                else:
                    filename = TF2_WEIGHTS_NAME

                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "resume_download": resume_download,
                        "local_files_only": local_files_only,
                        "token": token,
                        "user_agent": user_agent,
                        "revision": revision,
                        "subfolder": subfolder,
                        "_raise_exceptions_for_missing_entries": False,
                        "_commit_hash": commit_hash,
                    }
                    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == SAFE_WEIGHTS_NAME:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME, **cached_file_kwargs
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                            raise NotImplementedError(
                                "Support for sharded checkpoints using safetensors is coming soon!"
                            )
                        else:
                            # This repo has no safetensors file of any kind, we switch to TensorFlow.
                            filename = TF2_WEIGHTS_NAME
                            resolved_archive_file = cached_file(
                                pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **cached_file_kwargs
                            )
                    if resolved_archive_file is None and filename == TF2_WEIGHTS_NAME:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, TF2_WEIGHTS_INDEX_NAME, **cached_file_kwargs
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None and filename == WEIGHTS_NAME:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, WEIGHTS_INDEX_NAME, **cached_file_kwargs
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None:
                        # Otherwise, maybe there is a PyTorch or Flax model file.  We try those to give a helpful error
                        # message.
                        has_file_kwargs = {
                            "revision": revision,
                            "proxies": proxies,
                            "token": token,
                        }
                        if has_file(pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {TF2_WEIGHTS_NAME} but there is a file for PyTorch weights. Use `from_pt=True` to"
                                " load this model from those weights."
                            )
                        else:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME},"
                                f" {TF2_WEIGHTS_NAME} or {TF_WEIGHTS_NAME}"
                            )

                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception:
                    # For any other exception, we throw a generic error.

                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME} or {TF_WEIGHTS_NAME}"
                    )
            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
                filename = resolved_archive_file.split(os.path.sep)[-1]
            else:
                logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, _ = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                _commit_hash=commit_hash,
            )

        safetensors_from_pt = False
        if filename == SAFE_WEIGHTS_NAME:
            with safe_open(resolved_archive_file, framework="tf") as f:
                safetensors_metadata = f.metadata()
            if safetensors_metadata is None or safetensors_metadata.get("format") not in ["pt", "tf", "flax"]:
                raise OSError(
                    f"The safetensors archive passed at {resolved_archive_file} does not contain the valid metadata."
                    " Make sure you save your model with the `save_pretrained` method."
                )
            safetensors_from_pt = safetensors_metadata.get("format") == "pt"

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
            return load_pytorch_checkpoint_in_tf2_model(
                model,
                resolved_archive_file,
                allow_missing_keys=True,
                output_loading_info=output_loading_info,
                _prefix=load_weight_prefix,
                tf_to_pt_weight_rename=tf_to_pt_weight_rename,
            )

        # we might need to extend the variable scope for composite models
        if load_weight_prefix is not None:
            with tf.compat.v1.variable_scope(load_weight_prefix):
                model.build()  # build the network with dummy inputs
        else:
            model.build()  # build the network with dummy inputs

        if safetensors_from_pt:

            with safe_open(resolved_archive_file, framework="tf") as safetensors_archive:
                # Load from a PyTorch checkpoint
                # We load in TF format here because PT weights often need to be transposed, and this is much
                # faster on GPU. Loading as numpy and transposing on CPU adds several seconds to load times.
                return load_pytorch_state_dict_in_keras_core_model(
                    model,
                    safetensors_archive,
                    tf_inputs=False,  # No need to build the model again
                    allow_missing_keys=True,
                    output_loading_info=output_loading_info,
                    _prefix=load_weight_prefix,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                )
        # 'by_name' allow us to do transfer learning by skipping/adding layers
        # see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1339-L1357
        try:
            if is_sharded:
                # TODO Matt: Sharded weights should probably be supported for Keras Core right now
                raise ValueError("Sharded weights are not supported for Keras Core right now!")
            else:
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

        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some layers from the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.warning(f"All model checkpoint layers were used when initializing {model.__class__.__name__}.\n")

        if len(missing_keys) > 0:
            logger.warning(
                f"Some layers of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.warning(
                f"All the layers of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **kwargs,
                )
            except OSError:
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )
                pass

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
            }

            return model, loading_info

        return model

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        max_shard_size: Optional[Union[int, str]] = "10GB",
        token: Optional[Union[bool, str]] = None,
        # (`use_auth_token` is deprecated: we have to keep it here as we don't have **kwargs)
        use_auth_token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
        **base_model_card_args,
    ) -> str:
        """
        Upload the model files to the 🤗 Model Hub while synchronizing a local clone of the repo in `repo_path_or_name`.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your model to. It should contain your organization name
                when pushing to a given organization.
            use_temp_dir (`bool`, *optional*):
                Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload model"`.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
                Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
                will then be each of size lower than this size. If expressed as a string, needs to be digits followed
                by a unit (like `"5MB"`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.

        Examples:

        ```python
        from transformers import TFAutoModel

        model = TFAutoModel.from_pretrained("bert-base-cased")

        # Push the model to your namespace with the name "my-finetuned-bert".
        model.push_to_hub("my-finetuned-bert")

        # Push the model to an organization with the name "my-finetuned-bert".
        model.push_to_hub("huggingface/my-finetuned-bert")
        ```
        """
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if "repo_path_or_name" in base_model_card_args:
            warnings.warn(
                "The `repo_path_or_name` argument is deprecated and will be removed in v5 of Transformers. Use "
                "`repo_id` instead."
            )
            repo_id = base_model_card_args.pop("repo_path_or_name")
        # Deprecation warning will be sent after for repo_url and organization
        repo_url = base_model_card_args.pop("repo_url", None)
        organization = base_model_card_args.pop("organization", None)

        if os.path.isdir(repo_id):
            working_dir = repo_id
            repo_id = repo_id.split(os.path.sep)[-1]
        else:
            working_dir = repo_id.split("/")[-1]

        repo_id = self._create_repo(
            repo_id, private=private, token=token, repo_url=repo_url, organization=organization
        )

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            self.save_pretrained(work_dir, max_shard_size=max_shard_size)
            if hasattr(self, "history") and hasattr(self, "create_model_card"):
                # This is a Keras model and we might be able to fish out its History and make a model card out of it
                base_model_card_args = {
                    "output_dir": work_dir,
                    "model_name": Path(repo_id).name,
                }
                base_model_card_args.update(base_model_card_args)
                self.create_model_card(**base_model_card_args)

            self._upload_modified_files(
                work_dir,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
            )

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


def load_tf_weights_from_safetensors(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    # Read the safetensors file
    with safe_open(resolved_archive_file, framework="tf") as safetensors_archive:
        mismatched_layers = []
        weight_names = [format_weight_name(w.path, _prefix=_prefix) for w in model.weights]
        loaded_weight_names = list(safetensors_archive.keys())
        # Find the missing layers from the high level list of layers
        missing_layers = list(set(weight_names) - set(loaded_weight_names))
        # Find the unexpected layers from the high level list of layers
        unexpected_layers = list(set(loaded_weight_names) - set(weight_names))

        for weight in model.weights:
            weight_name = format_weight_name(weight.name, _prefix=_prefix)
            if weight_name in loaded_weight_names:
                weight_value = safetensors_archive.get_tensor(weight_name)
                # Check if the shape of the current weight and the one from the H5 file are different
                if K.int_shape(weight) != weight_value.shape:
                    # If yes we reshape the weight from the H5 file accordingly to the current weight
                    # If the two shapes are not compatible we raise an issue
                    try:
                        weight_value = tf.reshape(weight_value, K.int_shape(weight))
                    except ValueError as e:
                        if ignore_mismatched_sizes:
                            mismatched_layers.append((weight_name, weight_value.shape, K.int_shape(weight)))
                            continue
                        else:
                            raise e

                K.set_value(weight, weight_value)  # weight.assign() might break if weight is a DTensor
    return missing_layers, unexpected_layers, mismatched_layers

def load_pytorch_state_dict_in_keras_core_model(
    tf_model,
    pt_state_dict,
    tf_inputs=None,
    allow_missing_keys=False,
    output_loading_info=False,
    _prefix=None,
    tf_to_pt_weight_rename=None,
    ignore_mismatched_sizes=False,
):
    from .modeling_tf_pytorch_utils import convert_tf_weight_name_to_pt_weight_name, apply_transpose, tensor_size
    """Load a pytorch state_dict in a TF 2.0 model. pt_state_dict can be either an actual dict or a lazy-loading
    safetensors archive created with the safe_open() function."""
    import tensorflow as tf
    from keras import backend as K

    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs
    if _prefix is None:
        _prefix = ""
    if tf_inputs:
        with tf.name_scope(_prefix):
            tf_model(tf_inputs, training=False)  # Make sure model is built
    # Convert old format to new format if needed from a PyTorch state_dict
    tf_keys_to_pt_keys = {}
    for key in pt_state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if "running_var" in key:
            new_key = key.replace("running_var", "moving_variance")
        if "running_mean" in key:
            new_key = key.replace("running_mean", "moving_mean")

        # New `weight_norm` from https://github.com/huggingface/transformers/pull/24030
        key_components = key.split(".")
        name = None
        if key_components[-3::2] == ["parametrizations", "original0"]:
            name = key_components[-2] + "_g"
        elif key_components[-3::2] == ["parametrizations", "original1"]:
            name = key_components[-2] + "_v"
        if name is not None:
            key_components = key_components[:-3] + [name]
            new_key = ".".join(key_components)

        if new_key is None:
            new_key = key
        tf_keys_to_pt_keys[new_key] = key

    # Matt: All TF models store the actual model stem in a MainLayer class, including the base model.
    # In PT, the derived models (with heads) use the base model class as the stem instead,
    # and there is no MainLayer class. This means that TF base classes have one
    # extra layer in their weight names, corresponding to the MainLayer class. This code block compensates for that.
    start_prefix_to_remove = ""
    if not any(s.startswith(tf_model.base_model_prefix) for s in tf_keys_to_pt_keys.keys()):
        start_prefix_to_remove = tf_model.base_model_prefix + "."

    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    tf_loaded_numel = 0
    all_pytorch_weights = set(tf_keys_to_pt_keys.keys())
    missing_keys = []
    mismatched_keys = []
    is_safetensor_archive = hasattr(pt_state_dict, "get_tensor")
    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.path
        name, transpose = convert_tf_weight_name_to_pt_weight_name(
            sw_name,
            start_prefix_to_remove=start_prefix_to_remove,
            tf_weight_shape=symbolic_weight.shape,
            name_scope=_prefix,
        )
        if tf_to_pt_weight_rename is not None:
            name = tf_to_pt_weight_rename(name)

        # Find associated numpy array in pytorch model state dict
        if name not in tf_keys_to_pt_keys:
            if allow_missing_keys:
                missing_keys.append(name)
                continue
            elif tf_model._keys_to_ignore_on_load_missing is not None:
                # authorized missing keys don't have to be loaded
                if any(re.search(pat, name) is not None for pat in tf_model._keys_to_ignore_on_load_missing):
                    continue
            raise AttributeError(f"{name} not found in PyTorch model")
        state_dict_name = tf_keys_to_pt_keys[name]
        if is_safetensor_archive:
            array = pt_state_dict.get_tensor(state_dict_name)
        else:
            array = pt_state_dict[state_dict_name]
        try:
            array = apply_transpose(transpose, array, symbolic_weight.shape)
        except tf.errors.InvalidArgumentError as e:
            if not ignore_mismatched_sizes:
                error_msg = str(e)
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
                raise tf.errors.InvalidArgumentError(error_msg)
            else:
                mismatched_keys.append((name, array.shape, symbolic_weight.shape))
                continue

        tf_loaded_numel += tensor_size(array)

        K.set_value(symbolic_weight, array)
        del array  # Immediately free memory to keep peak usage as low as possible
        all_pytorch_weights.discard(name)

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

    if len(mismatched_keys) > 0:
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        logger.warning(
            f"Some weights of {tf_model.__class__.__name__} were not initialized from the model checkpoint"
            f" are newly initialized because the shapes did not"
            f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
            " to use it for predictions and inference."
        )

    if output_loading_info:
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
        }
        return tf_model, loading_info

    return tf_model
