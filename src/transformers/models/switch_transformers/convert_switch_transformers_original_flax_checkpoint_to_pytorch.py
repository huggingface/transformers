# Copyright 2022 The HuggingFace Inc. team.
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

"""Convert SwitchTransformersX checkpoints from the original repository to JAX/FLAX model."""

import argparse
import re

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from t5x import checkpoints

from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration
from transformers.utils import logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def load_flax_weights_in_pytorch_model(pt_model, flax_state):
    """Load flax checkpoints in a PyTorch model"""

    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        logger.error(
            "Loading a Flax weights in PyTorch, requires both PyTorch and Flax to be installed. Please see"
            " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/index.html#installation for installation"
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
        pt_model.base_model_prefix not in {k.split(".")[0] for k in pt_model_dict}
    )
    load_base_model_into_model_with_head = (pt_model.base_model_prefix not in flax_state) and (
        pt_model.base_model_prefix in {k.split(".")[0] for k in pt_model_dict}
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

        # adding batch stats from flax batch norm to pt
        elif "mean" in flax_key_tuple[-1]:
            flax_key_tuple = flax_key_tuple[:-1] + ("running_mean",)
        elif "var" in flax_key_tuple[-1]:
            flax_key_tuple = flax_key_tuple[:-1] + ("running_var",)

        if "batch_stats" in flax_state:
            flax_key = ".".join(flax_key_tuple[1:])  # Remove the params/batch_stats header
        else:
            flax_key = ".".join(flax_key_tuple)

        # We also need to look at `pt_model_dict` and see if there are keys requiring further transformation.
        special_pt_names = {}
        # New `weight_norm` from https://github.com/huggingface/transformers/pull/24030
        for key in pt_model_dict:
            key_components = key.split(".")
            name = None
            if key_components[-3::2] == ["parametrizations", "original0"]:
                name = key_components[-2] + "_g"
            elif key_components[-3::2] == ["parametrizations", "original1"]:
                name = key_components[-2] + "_v"
            if name is not None:
                key_components = key_components[:-3] + [name]
                key_to_check = ".".join(key_components)
                special_pt_names[key_to_check] = key

        if flax_key in special_pt_names:
            flax_key = special_pt_names[flax_key]

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


# should not include what is already done by the `from_pt` argument
MOE_LAYER_NAME_MAPPING = {
    "/attention/": "/0/SelfAttention/",
    "/self_attention/": "/0/SelfAttention/",
    "/encoder_decoder_attention/": "/1/EncDecAttention/",
    "value": "v",
    "query": "q",
    "key": "k",
    "out": "o",
    "pre_self_attention_layer_norm": "0/layer_norm",
    "pre_cross_attention_layer_norm": "1/layer_norm",
    "pre_attention_layer_norm": "0/layer_norm",  # previously 1, but seems wrong
    "token_embedder": "shared",
    "encoder_norm": "final_layer_norm",
    "decoder_norm": "final_layer_norm",
    "relpos_bias/rel_embedding": "block/0/layer/0/SelfAttention/relative_attention_bias/weight",
    "router/router_weights/w/": "router/classifier/",
    "roer/roer_weights/w/": "router/classifier/",
    "logits_dense": "lm_head",
}


def rename_keys(s_dict):
    # 1. in HF T5, we have block.{x}.layer.{y}. which corresponds to layer.{x} in
    # the original model
    keys = list(s_dict.keys())
    for key in keys:
        layer_to_block_of_layer = r".*/layers_(\d+)"
        new_key = key
        if re.match(layer_to_block_of_layer, key):
            new_key = re.sub(r"layers_(\d+)", r"block/\1/layer", new_key)

        layer_to_block_of_layer = r"(encoder|decoder)\/"

        if re.match(layer_to_block_of_layer, key):
            groups = re.match(layer_to_block_of_layer, new_key).groups()
            if groups[0] == "encoder":
                new_key = re.sub(r"/mlp/", r"/1/mlp/", new_key)
                new_key = re.sub(r"/pre_mlp_layer_norm/", r"/1/layer_norm/", new_key)

            elif groups[0] == "decoder":
                new_key = re.sub(r"/mlp/", r"/2/mlp/", new_key)
                new_key = re.sub(r"/pre_mlp_layer_norm/", r"/2/layer_norm/", new_key)

        # 2. Convert other classic mappings
        for old_key, temp_key in MOE_LAYER_NAME_MAPPING.items():
            if old_key in new_key:
                new_key = new_key.replace(old_key, temp_key)

        print(f"{key} -> {new_key}")
        s_dict[new_key] = s_dict.pop(key)

    if "encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight" in s_dict:
        s_dict["encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"] = s_dict[
            "encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"
        ].T
    if "decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight" in s_dict:
        s_dict["decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"] = s_dict[
            "decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"
        ].T

    # 3. Take extra care of the EXPERTS layer
    for key in list(s_dict.keys()):
        if "expert" in key:
            num_experts = s_dict[key].shape[0]
            expert_weihts = s_dict[key]
            for idx in range(num_experts):
                s_dict[key.replace("expert/", f"experts/expert_{idx}/")] = expert_weihts[idx]
                print(f"{key} -> {key.replace('expert/', f'experts/expert_{idx}/')}")

            s_dict.pop(key)

    return s_dict


GIN_TO_CONFIG_MAPPING = {
    "NUM_ENCODER_LAYERS": "num_layers",
    "NUM_DECODER_LAYERS": "num_decoder_layers",
    "NUM_HEADS": "num_heads",
    "HEAD_DIM": "d_kv",
    "EMBED_DIM": "d_model",
    "MLP_DIM": "d_ff",
    "NUM_SELECTED_EXPERTS": "num_selected_experts",
    "NUM_ENCODER_SPARSE_LAYERS": "num_sparse_encoder_layers",
    "NUM_DECODER_SPARSE_LAYERS": "num_sparse_decoder_layers",
    "dense.MlpBlock.activations": "feed_forward_proj",
}


def convert_gin_to_config(gin_file, num_experts):
    # Convert a google style config to the hugging face format
    with open(gin_file, "r") as f:
        raw_gin = f.read()

    regex_match = re.findall(r"(.*) = ([0-9.]*)", raw_gin)
    args = {}
    for param, value in regex_match:
        if param in GIN_TO_CONFIG_MAPPING and value != "":
            args[GIN_TO_CONFIG_MAPPING[param]] = float(value) if "." in value else int(value)

    activation = re.findall(r"(.*activations) = \(\'(.*)\',\)", raw_gin)[0]
    args[GIN_TO_CONFIG_MAPPING[activation[0]]] = str(activation[1])

    args["num_experts"] = num_experts
    config = SwitchTransformersConfig(**args)
    return config


def convert_flax_checkpoint_to_pytorch(
    flax_checkpoint_path, config_file, gin_file=None, pytorch_dump_path="./", num_experts=8
):
    # Initialise PyTorch model

    print(f"Loading flax weights from : {flax_checkpoint_path}")
    flax_params = checkpoints.load_t5x_checkpoint(flax_checkpoint_path)

    if gin_file is not None:
        config = convert_gin_to_config(gin_file, num_experts)
    else:
        config = SwitchTransformersConfig.from_pretrained(config_file)

    pt_model = SwitchTransformersForConditionalGeneration(config)

    flax_params = flax_params["target"]
    flax_params = flatten_dict(flax_params, sep="/")
    flax_params = rename_keys(flax_params)
    flax_params = unflatten_dict(flax_params, sep="/")

    # Load the flax params in the PT model
    load_flax_weights_in_pytorch_model(pt_model, flax_params)

    print(f"Save PyTorch model to {pytorch_dump_path}")
    pt_model.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--switch_t5x_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained SwitchTransformers model. \nThis specifies the"
            " model architecture. If not provided, a `gin_file` has to be provided."
        ),
    )
    parser.add_argument(
        "--gin_file",
        default=None,
        type=str,
        required=False,
        help="Path to the gin config file. If not provided, a `config_file` has to be passed   ",
    )
    parser.add_argument(
        "--config_name", default=None, type=str, required=False, help="Config name of SwitchTransformers model."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output pytorch model."
    )
    parser.add_argument("--num_experts", default=8, type=int, required=False, help="Number of experts")
    args = parser.parse_args()
    convert_flax_checkpoint_to_pytorch(
        args.switch_t5x_checkpoint_path,
        args.config_name,
        args.gin_file,
        args.pytorch_dump_folder_path,
        args.num_experts,
    )
