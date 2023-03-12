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
"""Convert BigScience BLOOM checkpoint."""


import argparse
import json
import os
import re

import torch

from transformers import BloomConfig, BloomModel
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging


logging.set_verbosity_info()

WEIGHTS_TO_AVERAGE_ENDSWITH = [
    "word_embeddings_layernorm.weight",
    "word_embeddings_layernorm.bias",
    "input_layernorm.weight",
    "input_layernorm.bias",
    "post_attention_layernorm.weight",
    "post_attention_layernorm.bias",
    "self_attention.dense.bias",
    "mlp.dense_4h_to_h.bias",
    "ln_f.weight",
    "ln_f.bias",
]

WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN = [
    "mlp.dense_4h_to_h.weight",
    "self_attention.dense.weight",
]


def layer_name_mapping(key, file):
    """Convert Megatron-DeepSpeed TP/PP weights mapping in transformers PP only"""
    # Handle first and last layers
    layer_rename_map = {
        "word_embeddings.weight": "word_embeddings.weight",
        "word_embeddings.norm.weight": "word_embeddings_layernorm.weight",
        "word_embeddings.norm.bias": "word_embeddings_layernorm.bias",
        "weight": "ln_f.weight",
        "bias": "ln_f.bias",
    }

    if key in layer_rename_map:
        return layer_rename_map[key]

    # Handle transformer blocks
    layer_number = int(re.match(r".*layer_(\d*).*", file)[1])
    layer_number -= 3
    return f"h.{layer_number}." + key


def get_dtype_size(dtype):
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search("[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def convert_bloom_checkpoint_to_pytorch(
    bloom_checkpoint_path, bloom_config_file, pytorch_dump_folder_path, shard_model, pretraining_tp
):
    # Construct model
    if bloom_config_file == "":
        config = BloomConfig()
    else:
        config = BloomConfig.from_json_file(bloom_config_file)

    if shard_model:
        file_names = os.listdir(bloom_checkpoint_path)
        file_names = sorted(filter(lambda s: s.startswith("layer") and "model_00" in s, file_names))

        index_dict = {"weight_map": {}, "metadata": {}}
        total_size = 0

        missing_keys = None

        config = BloomConfig()

        for j, file in enumerate(file_names):
            print("Processing file: {}".format(file))
            tensors = None

            for i in range(pretraining_tp):
                # load all TP files
                f_name = file.replace("model_00", f"model_0{i}")
                temp = torch.load(os.path.join(bloom_checkpoint_path, f_name), map_location="cpu")

                # Rename keys in the transformers names
                keys = list(temp.keys())
                for key in keys:
                    temp[layer_name_mapping(key, file)] = temp.pop(key)

                if tensors is None:
                    tensors = temp
                else:
                    for key in tensors.keys():
                        if any(key.endswith(end) for end in WEIGHTS_TO_AVERAGE_ENDSWITH):
                            # We average (sum and then divide) some weights accross TP ranks (see https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/olruwase/sync_layer_norms/megatron/training.py#L425)
                            tensors[key] += temp[key]
                        else:
                            # Some weights are RowParallelLinear in Megatron-Deepspeed, others are ColumnParallel
                            cat_dim = 1 if any(text in key for text in WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN) else 0
                            # We concatenate these weights accross TP ranks
                            tensors[key] = torch.cat([tensors[key], temp[key]], dim=cat_dim)

            # Divide by the number of TP the weights we want to average
            for key in tensors.keys():
                if any(key.endswith(end) for end in WEIGHTS_TO_AVERAGE_ENDSWITH):
                    tensors[key] = tensors[key] / pretraining_tp
            torch.save(
                tensors,
                os.path.join(
                    pytorch_dump_folder_path,
                    "pytorch_model_{}-of-{}.bin".format(str(j + 1).zfill(5), str(len(file_names)).zfill(5)),
                ),
            )

            for key in tensors.keys():
                value = tensors[key]
                total_size += value.numel() * get_dtype_size(value.dtype)
                if key not in index_dict["weight_map"]:
                    index_dict["weight_map"][key] = "pytorch_model_{}-of-{}.bin".format(
                        str(j + 1).zfill(5), str(len(file_names)).zfill(5)
                    )

        config = BloomConfig()
        pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME
        index_dict["metadata"]["total_size"] = total_size
        with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
            f.write(config.to_json_string())
        with open(os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME + ".index.json"), "w", encoding="utf-8") as f:
            json_config = json.dumps(index_dict, indent=2, sort_keys=True) + "\n"
            f.write(json_config)
    else:
        model = BloomModel(config)

        file_names = os.listdir(bloom_checkpoint_path)
        file_names = sorted(filter(lambda s: s.startswith("layer") and "model_00" in s, file_names))

        missing_keys = None
        for i, file in enumerate(file_names):
            tensors = None
            for i in range(pretraining_tp):
                # load all TP files
                f_name = file.replace("model_00", f"model_0{i}")
                temp = torch.load(os.path.join(bloom_checkpoint_path, f_name), map_location="cpu")

                # Rename keys in the transformers names
                keys = list(temp.keys())
                for key in keys:
                    temp[layer_name_mapping(key, file)] = temp.pop(key)

                if tensors is None:
                    tensors = temp
                else:
                    for key in tensors.keys():
                        # We average (sum and then divide) some weights accross TP ranks (see https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/olruwase/sync_layer_norms/megatron/training.py#L425)
                        if any(key.endswith(end) for end in WEIGHTS_TO_AVERAGE_ENDSWITH):
                            tensors[key] += temp[key]
                        else:
                            # Some weights are RowParallelLinear in Megatron-Deepspeed, others are ColumnParallel
                            cat_dim = 1 if any(text in key for text in WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN) else 0
                            # We concatenate these weights accross TP ranks
                            tensors[key] = torch.cat([tensors[key], temp[key]], dim=cat_dim)

            # Divide by the number of TP the weights we want to average
            for key in tensors.keys():
                if any(key.endswith(end) for end in WEIGHTS_TO_AVERAGE_ENDSWITH):
                    tensors[key] = tensors[key] / pretraining_tp

            other_keys = model.load_state_dict(tensors, strict=False)
            assert not other_keys.unexpected_keys, f"The keys {other_keys.unexpected_keys} are unexpected"
            if missing_keys is None:
                missing_keys = set(other_keys.missing_keys)
            else:
                missing_keys = missing_keys.intersection(set(other_keys.missing_keys))

        assert not missing_keys, f"The keys {missing_keys} are missing"

        # Save pytorch-model
        os.makedirs(pytorch_dump_folder_path, exist_ok=True)
        pytorch_weights_dump_path = pytorch_dump_folder_path + "/" + WEIGHTS_NAME
        pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME
        print(f"Save PyTorch model to {pytorch_weights_dump_path} with dtype {config.torch_dtype}")
        if config.torch_dtype is not None:
            model = model.to(config.torch_dtype)
        torch.save(model.state_dict(), pytorch_weights_dump_path)
        print(f"Save configuration file to {pytorch_config_dump_path}")
        with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
            f.write(config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--bloom_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the Megatron-LM checkpoint path.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--bloom_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--shard_model",
        action="store_true",
        help="An optional setting to shard the output model \nThis enables sharding the converted checkpoint",
    )
    parser.add_argument(
        "--pretraining_tp",
        default=4,
        type=int,
        help="Pretraining TP rank that has been used when training the model in Megatron-LM \n",
    )
    args = parser.parse_args()
    convert_bloom_checkpoint_to_pytorch(
        args.bloom_checkpoint_path,
        args.bloom_config_file,
        args.pytorch_dump_folder_path,
        args.shard_model,
        args.pretraining_tp,
    )
