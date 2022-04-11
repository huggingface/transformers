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
"""Convert OpenAI GPT checkpoint."""


import argparse

import torch
import os
import re

from transformers import BigScience176BConfig, BigScience176BModel
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging


logging.set_verbosity_info()

WEIGHTS_TO_AVERAGE_ENDSWITH = [
        "word_embeddings_layernorm.weight",  # "word_embeddings.norm.weight",
        "word_embeddings_layernorm.bias",  # "word_embeddings.norm.bias",
        "input_layernorm.weight",
        "input_layernorm.bias",
        "post_attention_layernorm.weight",
        "post_attention_layernorm.bias",
        "self_attention.dense.bias",
        "mlp.dense_4h_to_h.bias",
        "ln_f.weight",
        "ln_f.bias"
]

WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN = [
    "dense_4h_to_h",
    "self_attention.dense.weight",
]

def layer_name_mapping(key, file):
    """ Convert Megatron-DeepSpeed TP/PP weights mapping in transformers PP only"""
    # Handle first and last layers
    if key == "word_embeddings.weight":
        return key
    if key == "word_embeddings.norm.weight":
        return "word_embeddings_layernorm.weight"
    if key == "word_embeddings.norm.bias":
        return "word_embeddings_layernorm.bias"
    if key == "weight":
        return "ln_f.weight"
    if key == "bias":
        return "ln_f.bias"

    # Handle transformer blocks
    layer_number = int(re.match(r'.*layer_(\d*).*', file)[1])
    layer_number -= 3
    return f'h.{layer_number}.' + key


def convert_bigscience176b_checkpoint_to_pytorch(bigscience176b_checkpoint_path, bigscience176b_config_file, pytorch_dump_folder_path):
    # Construct model
    if bigscience176b_config_file == "":
        config = BigScience176BConfig()
    else:
        config = BigScience176BConfig.from_json_file(bigscience176b_config_file)
    model = BigScience176BModel(config)
    print(model.state_dict().keys())

    file_names = os.listdir(bigscience176b_checkpoint_path)
    file_names = list(sorted(filter(lambda s: s.startswith('layer') and 'model_00' in s, file_names)))

    missing_keys = None
    for file in file_names:
        tensors = None
        for i in range(config.pretraining_tp):
            # load all TP files
            f_name = file.replace('model_00', f'model_0{i}')
            temp = torch.load(os.path.join(bigscience176b_checkpoint_path, f_name), map_location='cpu')

            # Rename keys in the transformers names
            keys = list(temp.keys())
            for key in keys:
                temp[layer_name_mapping(key, file)] = temp.pop(key)

            if tensors is None:
                tensors = temp
            else:
                for key in tensors.keys():
                    if any(key.endswith(end) for end in WEIGHTS_TO_AVERAGE_ENDSWITH):
                        tensors[key] += temp[key]  # We average (sum and then divide) some weights accross TP ranks (see https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/olruwase/sync_layer_norms/megatron/training.py#L425)
                    else:
                        cat_dim = 1 if any(text in key for text in WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN) else 0  # Some weights are RowParallelLinear in Megatron-Deepspeed, others are ColumnParallel
                        tensors[key] = torch.cat([tensors[key], temp[key]], dim=cat_dim)  # We concatenate these weights accross TP ranks

        # Divide by the number of TP the weights we want to average
        for key in tensors.keys():
            if any(key.endswith(end) for end in WEIGHTS_TO_AVERAGE_ENDSWITH):
                tensors[key] = tensors[key]/config.pretraining_tp

        other_keys = model.load_state_dict(tensors, strict=False)
        assert not other_keys.unexpected_keys
        if missing_keys is None:
            missing_keys = set(other_keys.missing_keys)
        else:
            missing_keys = missing_keys.intersection(set(other_keys.missing_keys))

    assert not missing_keys



    # Save pytorch-model
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    pytorch_weights_dump_path = pytorch_dump_folder_path + "/" + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME
    print(f"Save PyTorch model to {pytorch_weights_dump_path}")
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print(f"Save configuration file to {pytorch_config_dump_path}")
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--bigscience176b_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--bigscience176b_config_file",
        default="",
        type=str,
        help="An optional config json file corresponding to the pre-trained model. \n"
        "This specifies the model architecture.",
    )
    args = parser.parse_args()
    convert_bigscience176b_checkpoint_to_pytorch(args.bigscience176b_checkpoint_path, args.bigscience176b_config_file, args.pytorch_dump_folder_path)
