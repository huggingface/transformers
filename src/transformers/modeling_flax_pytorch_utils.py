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
""" PyTorch - TF 2.0 general utilities."""


import os

from flax.core.frozen_dict import unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from .utils import logging


logger = logging.get_logger(__name__)


#####################
# PyTorch => Flax #
#####################


def load_pytorch_checkpoint_in_flax_state_dict(flax_model, pytorch_checkpoint_path, allow_missing_keys=False):
    """Load pytorch checkpoints in a flax model"""
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a PyTorch model in Flax, requires both PyTorch and  Flax to be installed. Please see "
            "https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation instructions."
        )
        raise

    pt_path = os.path.abspath(pytorch_checkpoint_path)
    logger.info(f"Loading PyTorch weights from {pt_path}")

    pt_state_dict = torch.load(pt_path, map_location="cpu")
    logger.info(f"PyTorch checkpoint contains {sum(t.numel() for t in pt_state_dict.values()):,} parameters.")

    flax_state_dict = convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)

    return flax_state_dict


def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model):
    # convert pytorch tensor to numpy
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

    random_flax_state_dict = flatten_dict(unfreeze(flax_model.params))
    flax_state_dict = {}

    remove_base_model_prefix = (flax_model.base_model_prefix not in flax_model.params) and (
        flax_model.base_model_prefix in set([k.split(".")[0] for k in pt_state_dict.keys()])
    )
    add_base_model_prefix = (flax_model.base_model_prefix in flax_model.params) and (
        flax_model.base_model_prefix not in set([k.split(".")[0] for k in pt_state_dict.keys()])
    )

    # Need to change some parameters name to match Flax names so that we don't have to fork any layer
    for pt_key, pt_tensor in pt_state_dict.items():

        pt_tuple_key = tuple(pt_key.split("."))

        has_base_model_prefix = pt_tuple_key[0] == flax_model.base_model_prefix
        require_base_model_prefix = (flax_model.base_model_prefix,) + pt_tuple_key in random_flax_state_dict

        if remove_base_model_prefix and has_base_model_prefix:
            pt_tuple_key = pt_tuple_key[1:]
        elif add_base_model_prefix and require_base_model_prefix:
            pt_tuple_key = (flax_model.base_model_prefix,) + pt_tuple_key

        if pt_tuple_key[-1] == "weight" and pt_tuple_key not in random_flax_state_dict:
            pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
            pt_tensor = pt_tensor.T
        elif pt_tuple_key[-1] == "gamma":
            pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
        elif pt_tuple_key[-1] == "beta":
            pt_tuple_key = pt_tuple_key[:-1] + ("bias",)

        if pt_tuple_key in random_flax_state_dict:
            if random_flax_state_dict[pt_tuple_key].shape != pt_tensor.shape:
                raise ValueError(
                    "PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape {random_flax_state_dict[pt_tuple_key].shape}, but is {pt_tensor.shape}."
                )

        # add unexpected weight so that warning is thrown
        flax_state_dict[pt_tuple_key] = pt_tensor

    return unflatten_dict(flax_state_dict)
