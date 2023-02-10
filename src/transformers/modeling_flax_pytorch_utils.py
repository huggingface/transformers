# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
""" PyTorch - Flax general utilities."""


import os
from pickle import UnpicklingError
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes
from flax.traverse_util import flatten_dict, unflatten_dict

import transformers

from .utils import logging


logger = logging.get_logger(__name__)


#####################
# PyTorch => Flax #
#####################


def load_pytorch_checkpoint_in_flax_state_dict(
    flax_model, pytorch_checkpoint_path, is_sharded, allow_missing_keys=False
):
    """Load pytorch checkpoints in a flax model"""
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a PyTorch model in Flax, requires both PyTorch and Flax to be installed. Please see"
            " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation"
            " instructions."
        )
        raise

    if not is_sharded:
        pt_path = os.path.abspath(pytorch_checkpoint_path)
        logger.info(f"Loading PyTorch weights from {pt_path}")

        pt_state_dict = torch.load(pt_path, map_location="cpu")
        logger.info(f"PyTorch checkpoint contains {sum(t.numel() for t in pt_state_dict.values()):,} parameters.")

        flax_state_dict = convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)
    else:
        # model is sharded and pytorch_checkpoint_path already contains the list of .pt shard files
        flax_state_dict = convert_pytorch_sharded_state_dict_to_flax(pytorch_checkpoint_path, flax_model)
    return flax_state_dict


def rename_key_and_reshape_tensor(
    pt_tuple_key: Tuple[str],
    pt_tensor: np.ndarray,
    random_flax_state_dict: Dict[str, jnp.ndarray],
    model_prefix: str,
) -> (Tuple[str], np.ndarray):
    """Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary"""

    def is_key_or_prefix_key_in_dict(key: Tuple[str]) -> bool:
        """Checks if `key` of `(prefix,) + key` is in random_flax_state_dict"""
        return len(set(random_flax_state_dict) & set([key, (model_prefix,) + key])) > 0

    # layer norm
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    if pt_tuple_key[-1] in ["weight", "gamma"] and is_key_or_prefix_key_in_dict(renamed_pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # embedding
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
    if pt_tuple_key[-1] == "weight" and is_key_or_prefix_key_in_dict(renamed_pt_tuple_key):
        return renamed_pt_tuple_key, pt_tensor

    # conv layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4 and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        return renamed_pt_tuple_key, pt_tensor

    # linear layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        pt_tensor = pt_tensor.T
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm weight
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
    if pt_tuple_key[-1] == "gamma":
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm bias
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
    if pt_tuple_key[-1] == "beta":
        return renamed_pt_tuple_key, pt_tensor

    return pt_tuple_key, pt_tensor


def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model):
    # convert pytorch tensor to numpy
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

    model_prefix = flax_model.base_model_prefix
    random_flax_state_dict = flatten_dict(flax_model.params)
    flax_state_dict = {}

    load_model_with_head_into_base_model = (model_prefix not in flax_model.params) and (
        model_prefix in set([k.split(".")[0] for k in pt_state_dict.keys()])
    )
    load_base_model_into_model_with_head = (model_prefix in flax_model.params) and (
        model_prefix not in set([k.split(".")[0] for k in pt_state_dict.keys()])
    )

    # Need to change some parameters name to match Flax names
    for pt_key, pt_tensor in pt_state_dict.items():
        pt_tuple_key = tuple(pt_key.split("."))

        # remove base model prefix if necessary
        has_base_model_prefix = pt_tuple_key[0] == model_prefix
        if load_model_with_head_into_base_model and has_base_model_prefix:
            pt_tuple_key = pt_tuple_key[1:]

        # Correctly rename weight parameters
        flax_key, flax_tensor = rename_key_and_reshape_tensor(
            pt_tuple_key, pt_tensor, random_flax_state_dict, model_prefix
        )

        # add model prefix if necessary
        require_base_model_prefix = (model_prefix,) + flax_key in random_flax_state_dict
        if load_base_model_into_model_with_head and require_base_model_prefix:
            flax_key = (model_prefix,) + flax_key

        if flax_key in random_flax_state_dict:
            if flax_tensor.shape != random_flax_state_dict[flax_key].shape:
                raise ValueError(
                    f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                    f"{random_flax_state_dict[flax_key].shape}, but is {flax_tensor.shape}."
                )

        # also add unexpected weight so that warning is thrown
        flax_state_dict[flax_key] = jnp.asarray(flax_tensor)

    return unflatten_dict(flax_state_dict)


############################
# Sharded Pytorch => Flax #
############################


def convert_pytorch_sharded_state_dict_to_flax(shard_filenames, flax_model):
    import torch

    # Load the index
    flax_state_dict = {}
    for shard_file in shard_filenames:
        # load using msgpack utils
        pt_state_dict = torch.load(shard_file)
        pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

        model_prefix = flax_model.base_model_prefix
        random_flax_state_dict = flatten_dict(flax_model.params)

        load_model_with_head_into_base_model = (model_prefix not in flax_model.params) and (
            model_prefix in set([k.split(".")[0] for k in pt_state_dict.keys()])
        )
        load_base_model_into_model_with_head = (model_prefix in flax_model.params) and (
            model_prefix not in set([k.split(".")[0] for k in pt_state_dict.keys()])
        )
        # Need to change some parameters name to match Flax names
        for pt_key, pt_tensor in pt_state_dict.items():
            pt_tuple_key = tuple(pt_key.split("."))

            # remove base model prefix if necessary
            has_base_model_prefix = pt_tuple_key[0] == model_prefix
            if load_model_with_head_into_base_model and has_base_model_prefix:
                pt_tuple_key = pt_tuple_key[1:]

            # Correctly rename weight parameters
            flax_key, flax_tensor = rename_key_and_reshape_tensor(
                pt_tuple_key, pt_tensor, random_flax_state_dict, model_prefix
            )
            # add model prefix if necessary
            require_base_model_prefix = (model_prefix,) + flax_key in random_flax_state_dict
            if load_base_model_into_model_with_head and require_base_model_prefix:
                flax_key = (model_prefix,) + flax_key

            if flax_key in random_flax_state_dict:
                if flax_tensor.shape != random_flax_state_dict[flax_key].shape:
                    raise ValueError(
                        f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                        f"{random_flax_state_dict[flax_key].shape}, but is {flax_tensor.shape}."
                    )

            # also add unexpected weight so that warning is thrown
            flax_state_dict[flax_key] = jnp.asarray(flax_tensor)
    return unflatten_dict(flax_state_dict)


#####################
# Flax => PyTorch #
#####################


def load_flax_checkpoint_in_pytorch_model(model, flax_checkpoint_path):
    """Load flax checkpoints in a PyTorch model"""
    flax_checkpoint_path = os.path.abspath(flax_checkpoint_path)
    logger.info(f"Loading Flax weights from {flax_checkpoint_path}")

    # import correct flax class
    flax_cls = getattr(transformers, "Flax" + model.__class__.__name__)

    # load flax weight dict
    with open(flax_checkpoint_path, "rb") as state_f:
        try:
            flax_state_dict = from_bytes(flax_cls, state_f.read())
        except UnpicklingError:
            raise EnvironmentError(f"Unable to convert {flax_checkpoint_path} to Flax deserializable object. ")

    return load_flax_weights_in_pytorch_model(model, flax_state_dict)


def load_flax_weights_in_pytorch_model(pt_model, flax_state):
    """Load flax checkpoints in a PyTorch model"""

    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a Flax weights in PyTorch, requires both PyTorch and Flax to be installed. Please see"
            " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation"
            " instructions."
        )
        raise

    # check if we have bf16 weights
    is_type_bf16 = flatten_dict(jax.tree_util.tree_map(lambda x: x.dtype == jnp.bfloat16, flax_state)).values()
    if any(is_type_bf16):
        # convert all weights to fp32 if the are bf16 since torch.from_numpy can-not handle bf16
        # and bf16 is not fully supported in PT yet.
        logger.warning(
            "Found ``bfloat16`` weights in Flax model. Casting all ``bfloat16`` weights to ``float32`` "
            "before loading those in PyTorch model."
        )
        flax_state = jax.tree_util.tree_map(
            lambda params: params.astype(np.float32) if params.dtype == jnp.bfloat16 else params, flax_state
        )

    flax_state_dict = flatten_dict(flax_state)
    pt_model_dict = pt_model.state_dict()

    load_model_with_head_into_base_model = (pt_model.base_model_prefix in flax_state) and (
        pt_model.base_model_prefix not in set([k.split(".")[0] for k in pt_model_dict.keys()])
    )
    load_base_model_into_model_with_head = (pt_model.base_model_prefix not in flax_state) and (
        pt_model.base_model_prefix in set([k.split(".")[0] for k in pt_model_dict.keys()])
    )

    # keep track of unexpected & missing keys
    unexpected_keys = []
    missing_keys = set(pt_model_dict.keys())

    for flax_key_tuple, flax_tensor in flax_state_dict.items():
        has_base_model_prefix = flax_key_tuple[0] == pt_model.base_model_prefix
        require_base_model_prefix = ".".join((pt_model.base_model_prefix,) + flax_key_tuple) in pt_model_dict

        # adapt flax_key to prepare for loading from/to base model only
        if load_model_with_head_into_base_model and has_base_model_prefix:
            flax_key_tuple = flax_key_tuple[1:]
        elif load_base_model_into_model_with_head and require_base_model_prefix:
            flax_key_tuple = (pt_model.base_model_prefix,) + flax_key_tuple

        # rename flax weights to PyTorch format
        if flax_key_tuple[-1] == "kernel" and flax_tensor.ndim == 4 and ".".join(flax_key_tuple) not in pt_model_dict:
            # conv layer
            flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
            flax_tensor = jnp.transpose(flax_tensor, (3, 2, 0, 1))
        elif flax_key_tuple[-1] == "kernel" and ".".join(flax_key_tuple) not in pt_model_dict:
            # linear layer
            flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
            flax_tensor = flax_tensor.T
        elif flax_key_tuple[-1] in ["scale", "embedding"]:
            flax_key_tuple = flax_key_tuple[:-1] + ("weight",)

        flax_key = ".".join(flax_key_tuple)

        if flax_key in pt_model_dict:
            if flax_tensor.shape != pt_model_dict[flax_key].shape:
                raise ValueError(
                    f"Flax checkpoint seems to be incorrect. Weight {flax_key_tuple} was expected "
                    f"to be of shape {pt_model_dict[flax_key].shape}, but is {flax_tensor.shape}."
                )
            else:
                # add weight to pytorch dict
                flax_tensor = np.asarray(flax_tensor) if not isinstance(flax_tensor, np.ndarray) else flax_tensor
                pt_model_dict[flax_key] = torch.from_numpy(flax_tensor)
                # remove from missing keys
                missing_keys.remove(flax_key)
        else:
            # weight is not expected by PyTorch model
            unexpected_keys.append(flax_key)

    pt_model.load_state_dict(pt_model_dict)

    # re-transform missing_keys to list
    missing_keys = list(missing_keys)

    if len(unexpected_keys) > 0:
        logger.warning(
            "Some weights of the Flax model were not used when initializing the PyTorch model"
            f" {pt_model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing"
            f" {pt_model.__class__.__name__} from a Flax model trained on another task or with another architecture"
            " (e.g. initializing a BertForSequenceClassification model from a FlaxBertForPreTraining model).\n- This"
            f" IS NOT expected if you are initializing {pt_model.__class__.__name__} from a Flax model that you expect"
            " to be exactly identical (e.g. initializing a BertForSequenceClassification model from a"
            " FlaxBertForSequenceClassification model)."
        )
    else:
        logger.warning(f"All Flax model weights were used when initializing {pt_model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {pt_model.__class__.__name__} were not initialized from the Flax model and are newly"
            f" initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to"
            " use it for predictions and inference."
        )
    else:
        logger.warning(
            f"All the weights of {pt_model.__class__.__name__} were initialized from the Flax model.\n"
            "If your task is similar to the task the model of the checkpoint was trained on, "
            f"you can already use {pt_model.__class__.__name__} for predictions without further training."
        )

    return pt_model
