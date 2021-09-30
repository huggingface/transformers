# Copyright 2021 The HuggingFace Team. All rights reserved.
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

"""
This script can be used to convert a head-less TF 2.x MultiBERTs model to PyTorch, as published on the official GitHub:
https://github.com/tensorflow/models/tree/master/official/nlp/bert
"""

import argparse
import os

import tensorflow as tf
import torch

from transformers import BertConfig, BertForPreTraining
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def convert_multibert_checkpoint_to_pytorch(tf_checkpoint_path, config_path, save_path):
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")

    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    config = BertConfig.from_pretrained(config_path)
    model = BertForPreTraining(config)

    layer_nums = []
    for full_name, shape in init_vars:
        array = tf.train.load_variable(tf_path, full_name)
        names.append(full_name)
        split_names = full_name.split("/")
        for name in split_names:
            if name.startswith("layer_"):
                layer_nums.append(int(name.split("_")[-1]))

        arrays.append(array)
    logger.info(f"Read a total of {len(arrays):,} layers")

    name_to_array = dict(zip(names, arrays))

    # Check that number of layers match
    assert config.num_hidden_layers == len(list(set(layer_nums)))

    state_dict = model.state_dict()

    # Need to do this explicitly as it is a buffer
    position_ids = state_dict["bert.embeddings.position_ids"]
    new_state_dict = {"bert.embeddings.position_ids": position_ids}

    # Encoder Layers
    for weight_name in names:
        pt_weight_name = weight_name.replace("kernel", "weight").replace("gamma", "weight").replace("beta", "bias")
        name_split = pt_weight_name.split("/")
        for name_idx, name in enumerate(name_split):
            if name.startswith("layer_"):
                name_split[name_idx] = name.replace("_", ".")

        if name_split[-1].endswith("embeddings"):
            name_split.append("weight")

        if name_split[0] == "cls":
            if name_split[-1] == "output_bias":
                name_split[-1] = "bias"
            if name_split[-1] == "output_weights":
                name_split[-1] = "weight"

        if name_split[-1] == "weight" and name_split[-2] == "dense":
            name_to_array[weight_name] = name_to_array[weight_name].T

        pt_weight_name = ".".join(name_split)

        new_state_dict[pt_weight_name] = torch.from_numpy(name_to_array[weight_name])

    new_state_dict["cls.predictions.decoder.weight"] = new_state_dict["bert.embeddings.word_embeddings.weight"].clone()
    new_state_dict["cls.predictions.decoder.bias"] = new_state_dict["cls.predictions.bias"].clone().T
    # Load State Dict
    model.load_state_dict(new_state_dict)

    # Save PreTrained
    logger.info(f"Saving pretrained model to {save_path}")
    model.save_pretrained(save_path)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path",
        type=str,
        default="./seed_0/bert.ckpt",
        required=False,
        help="Path to the TensorFlow 2.x checkpoint path.",
    )
    parser.add_argument(
        "--bert_config_file",
        type=str,
        default="./bert_config.json",
        required=False,
        help="The config json file corresponding to the BERT model. This specifies the model architecture.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model (must include filename).",
    )
    args = parser.parse_args()

    convert_multibert_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.save_path)
