# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert Funnel checkpoint."""

import argparse
import os

import torch

from transformers import FunnelBaseModel, FunnelConfig, FunnelModel
from transformers.models.funnel.modeling_funnel import FunnelPositionwiseFFN, FunnelRelMultiheadAttention
from transformers.utils import logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def load_tf_weights_in_funnel(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    _layer_map = {
        "k": "k_head",
        "q": "q_head",
        "v": "v_head",
        "o": "post_proj",
        "layer_1": "linear_1",
        "layer_2": "linear_2",
        "rel_attn": "attention",
        "ff": "ffn",
        "kernel": "weight",
        "gamma": "weight",
        "beta": "bias",
        "lookup_table": "weight",
        "word_embedding": "word_embeddings",
        "input": "embeddings",
    }

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        if name[0] == "generator":
            continue
        pointer = model
        skipped = False
        for m_name in name[1:]:
            if not isinstance(pointer, FunnelPositionwiseFFN) and re.fullmatch(r"layer_\d+", m_name):
                layer_index = int(re.search(r"layer_(\d+)", m_name).groups()[0])
                if layer_index < config.num_hidden_layers:
                    block_idx = 0
                    while layer_index >= config.block_sizes[block_idx]:
                        layer_index -= config.block_sizes[block_idx]
                        block_idx += 1
                    pointer = pointer.blocks[block_idx][layer_index]
                else:
                    layer_index -= config.num_hidden_layers
                    pointer = pointer.layers[layer_index]
            elif m_name == "r" and isinstance(pointer, FunnelRelMultiheadAttention):
                pointer = pointer.r_kernel
                break
            elif m_name in _layer_map:
                pointer = getattr(pointer, _layer_map[m_name])
            else:
                try:
                    pointer = getattr(pointer, m_name)
                except AttributeError:
                    print(f"Skipping {'/'.join(name)}", array.shape)
                    skipped = True
                    break
        if not skipped:
            if len(pointer.shape) != len(array.shape):
                array = array.reshape(pointer.shape)
            if m_name == "kernel":
                array = np.transpose(array)
            pointer.data = torch.from_numpy(array)

    return model


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, base_model):
    # Initialise PyTorch model
    config = FunnelConfig.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")
    model = FunnelBaseModel(config) if base_model else FunnelModel(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_funnel(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained model. \nThis specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--base_model", action="store_true", help="Whether you want just the base model (no decoder) or not."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.base_model
    )
