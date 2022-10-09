# coding=utf-8
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
"""Convert DALLE_MEGA checkpoint."""


import numpy as np
import torch

import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict


try:
    from dalle_mini import DalleBart
except ImportError:
    print("dalle_mini is not installed. Please install it with `pip install dalle_mini`.")

from transformers import DalleMegaConfig, DalleMegaForConditionalGeneration


def convert_flax_state_dict_to_pt(flax_state):
    # convert all weights to fp32 if the are bf16 since torch.from_numpy can-not handle bf16
    # and bf16 is not fully supported in PT yet.
    flax_state = jax.tree_map(
        lambda params: params.astype(np.float32) if params.dtype == jnp.bfloat16 else params, flax_state
    )

    flax_state_dict = flatten_dict(flax_state)
    pt_model_dict = dict()

    for flax_key_tuple, flax_tensor in flax_state_dict.items():
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

        flax_tensor = np.asarray(flax_tensor) if not isinstance(flax_tensor, np.ndarray) else flax_tensor
        pt_model_dict[flax_key] = torch.from_numpy(flax_tensor)
    return pt_model_dict


replace_patterns = [
    ("GLU_0.Dense_0", "GLU_0.wi_0"),
    ("GLU_0.Dense_1", "GLU_0.wi_1"),
    ("GLU_0.Dense_2", "GLU_0.wo"),
    ("GLU_0.LayerNorm_", "GLU_0.layernorm_"),
    ("FlaxBartEncoderLayer_", ""),
    ("FlaxBartDecoderLayer_", ""),
    ("FlaxBartAttention_0", "self_attn"),
    ("FlaxBartAttention_1", "encoder_attn"),
    ("GLU_0", "ffn"),
    ("final_ln", "final_layernorm"),
    ("LayerNorm_0", "pre_attn_layernorm"),
    ("LayerNorm_1", "post_attn_layernorm"),
    ("LayerNorm_2", "pre_encoder_attn_layernorm"),
    ("LayerNorm_3", "post_encoder_attn_layernorm"),
]


def rename_key(key):
    for pattern in replace_patterns:
        key = key.replace(pattern[0], pattern[1])
    return key


def rename_state_dict(state_dict):
    keys = list(state_dict.keys())
    for key in keys:
        new_key = rename_key(key)
        state_dict[new_key] = state_dict.pop(key)
    return state_dict


def convert_dalle_mega_to_pt(checkpoint_path, config_path, save_path):
    _, params = DalleBart.from_pretrained(checkpoint_path, _do_init=False)

    config = DalleMegaConfig.from_pretrained(config_path)
    pt_model = DalleMegaForConditionalGeneration(config).eval()

    pt_state_dict = convert_flax_state_dict_to_pt(params)
    pt_state_dict = rename_state_dict(pt_state_dict)

    pt_model.load_state_dict(pt_state_dict)
    pt_model.save_pretrained(save_path)
    return pt_model
