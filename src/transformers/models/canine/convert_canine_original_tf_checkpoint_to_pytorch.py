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
"""Convert CANINE checkpoint."""

import argparse
import os

import torch

from transformers import CanineConfig, CanineModel, CanineTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def load_tf_weights_in_canine(model, config, tf_checkpoint_path):
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

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        # also discard the cls weights (which were used for the next sentence prediction pre-training task)
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
                "cls",
                "autoregressive_decoder",
                "char_output_weights",
            ]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        # if first scope name starts with "bert", change it to "encoder"
        if name[0] == "bert":
            name[0] = "encoder"
        # remove "embeddings" middle name of HashBucketCodepointEmbedders
        elif name[1] == "embeddings":
            name.remove(name[1])
        # rename segment_embeddings to token_type_embeddings
        elif name[1] == "segment_embeddings":
            name[1] = "token_type_embeddings"
        # rename initial convolutional projection layer
        elif name[1] == "initial_char_encoder":
            name = ["chars_to_molecules"] + name[-2:]
        # rename final convolutional projection layer
        elif name[0] == "final_char_encoder" and name[1] in ["LayerNorm", "conv"]:
            name = ["projection"] + name[1:]
        pointer = model
        for m_name in name:
            if (re.fullmatch(r"[A-Za-z]+_\d+", m_name)) and "Embedder" not in m_name:
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name[-10:] in [f"Embedder_{i}" for i in range(8)]:
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)

        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")

        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, pytorch_dump_path):
    # Initialize PyTorch model
    config = CanineConfig()
    model = CanineModel(config)
    model.eval()

    print(f"Building PyTorch model from configuration: {config}")

    # Load weights from tf checkpoint
    load_tf_weights_in_canine(model, config, tf_checkpoint_path)

    # Save pytorch-model (weights and configuration)
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

    # Save tokenizer files
    tokenizer = CanineTokenizer()
    print(f"Save tokenizer files to {pytorch_dump_path}")
    tokenizer.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint. Should end with model.ckpt",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to a folder where the PyTorch model will be placed.",
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.pytorch_dump_path)
