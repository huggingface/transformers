# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert ELECTRA checkpoint."""

import argparse
import os

import torch

from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining
from transformers.utils import logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def load_tf_weights_in_electra(model, config, tf_checkpoint_path, discriminator_or_generator="discriminator"):
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
        original_name: str = name

        try:
            if isinstance(model, ElectraForMaskedLM):
                name = name.replace("electra/embeddings/", "generator/embeddings/")

            if discriminator_or_generator == "generator":
                name = name.replace("electra/", "discriminator/")
                name = name.replace("generator/", "electra/")

            name = name.replace("dense_1", "dense_prediction")
            name = name.replace("generator_predictions/output_bias", "generator_lm_head/bias")

            name = name.split("/")
            # print(original_name, name)
            # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
            # which are not required for using pretrained model
            if any(n in ["global_step", "temperature"] for n in name):
                logger.info(f"Skipping {original_name}")
                continue
            pointer = model
            for m_name in name:
                if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                    scope_names = re.split(r"_(\d+)", m_name)
                else:
                    scope_names = [m_name]
                if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                    pointer = getattr(pointer, "bias")
                elif scope_names[0] == "output_weights":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "squad":
                    pointer = getattr(pointer, "classifier")
                else:
                    pointer = getattr(pointer, scope_names[0])
                if len(scope_names) >= 2:
                    num = int(scope_names[1])
                    pointer = pointer[num]
            if m_name.endswith("_embeddings"):
                pointer = getattr(pointer, "weight")
            elif m_name == "kernel":
                array = np.transpose(array)
            try:
                if pointer.shape != array.shape:
                    raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
            except ValueError as e:
                e.args += (pointer.shape, array.shape)
                raise
            print(f"Initialize PyTorch weight {name}", original_name)
            pointer.data = torch.from_numpy(array)
        except AttributeError as e:
            print(f"Skipping {original_name}", name, e)
            continue
    return model


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, discriminator_or_generator):
    # Initialise PyTorch model
    config = ElectraConfig.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")

    if discriminator_or_generator == "discriminator":
        model = ElectraForPreTraining(config)
    elif discriminator_or_generator == "generator":
        model = ElectraForMaskedLM(config)
    else:
        raise ValueError("The discriminator_or_generator argument should be either 'discriminator' or 'generator'")

    # Load weights from tf checkpoint
    load_tf_weights_in_electra(
        model, config, tf_checkpoint_path, discriminator_or_generator=discriminator_or_generator
    )

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
        "--discriminator_or_generator",
        default=None,
        type=str,
        required=True,
        help=(
            "Whether to export the generator or the discriminator. Should be a string, either 'discriminator' or "
            "'generator'."
        ),
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.discriminator_or_generator
    )
