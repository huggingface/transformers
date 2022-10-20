# coding=utf-8
# Copyright 2022 The SwitchTransformers authors and HuggingFace Inc. team.
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
"""Convert SwitchTransformers checkpoint."""


import argparse
import os

import regex as re
from flax.serialization import msgpack_restore
from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration
from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model
from transformers.utils import logging
from transformers.utils.hub import get_file_from_repo


logging.set_verbosity_info()

from flax.traverse_util import flatten_dict, unflatten_dict


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
    "pre_attention_layer_norm": "1/layer_norm",
    "token_embedder": "shared",
    "encoder_norm": "final_layer_norm",
    "decoder_norm": "final_layer_norm",
    "relpos_bias/rel_embedding": "block/0/layer/0/SelfAttention/relative_attention_bias/weight",
    "router/router_weights/w/": "router/classifier/",
    "roer/roer_weights/w/": "router/classifier/",
    
    

}

FLAX_MODELS = {
    "base-8": "https://huggingface.co/ybelkada/switch-c-base-8",
    "base-16": "https://huggingface.co/ybelkada/switch-c-base-16",
    "base-32": "https://huggingface.co/ybelkada/switch-c-base-32",
    "base-64": "https://huggingface.co/ybelkada/switch-c-base-64",
    "base-128": "https://huggingface.co/ybelkada/switch-c-base-128",
    "base-256": "https://huggingface.co/ybelkada/switch-c-base-256",
    "large-128": "https://huggingface.co/ybelkada/switch-c-large-128",
    "xxl-128": "https://huggingface.co/ybelkada/switch-c-xxl-128",
    "beast-2048": "https://huggingface.co/ybelkada/switch-c-2048",
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
            # s_dict[new_key] = s_dict.pop(key)

        layer_to_block_of_layer = r"(encoder|decoder)\/"

        if re.match(layer_to_block_of_layer, key):
            groups = re.match(layer_to_block_of_layer, new_key).groups()
            if groups[0] == "encoder":
                new_key = re.sub(r"/mlp/", r"/1/mlp/", new_key)
                new_key = re.sub(r"/pre_mlp_layer_norm/", r"/0/layer_norm/", new_key)

            elif groups[0] == "decoder":
                new_key = re.sub(r"/mlp/", r"/2/mlp/", new_key)
                new_key = re.sub(r"/pre_mlp_layer_norm/", r"/1/layer_norm/", new_key)

        # 2. Convert other classic mappings
        for old_key, temp_key in MOE_LAYER_NAME_MAPPING.items():
            if old_key in new_key:
                new_key = new_key.replace(old_key, temp_key)
                

        print(f"{key} -> {new_key}")
        s_dict[new_key] = s_dict.pop(key) 

        
    # 3. Take extra care of the EXPERTS layer
    for key in list(s_dict.keys()):
        if "expert" in key: 
            
            num_experts = s_dict[key].shape[0]
            expert_weihts = s_dict[key]
            for idx in range(num_experts):
                s_dict[key.replace("expert/", f"experts/expert_{idx}/")] = expert_weihts[idx]
            s_dict.pop(key)
    
    return s_dict


def convert_flax_checkpoint_to_pytorch(flax_checkpoint_path, config_file, pytorch_dump_path):
    # Initialise PyTorch model

    print(f"Loading flax weights from : {flax_checkpoint_path}")
    path = get_file_from_repo(flax_checkpoint_path, "flax_params.flax", use_auth_token = "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") # get_file_from_repo(config_file, "flax_params.flax") # get_file_from_repo(config_file, "flax_params.flax", use_auth_token = "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")
    
    config_file = get_file_from_repo(flax_checkpoint_path, "config.json", use_auth_token = "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") 
    with open(os.path.join(path), "rb") as f:
        params = msgpack_restore(f.read())

    config = SwitchTransformersConfig.from_pretrained(config_file, relative_attention_num_buckets=12)
    pt_model = SwitchTransformersForConditionalGeneration(config)

    params = flatten_dict(params, sep="/")
    params = rename_keys(params)
    params = unflatten_dict(params, sep="/")

    load_flax_weights_in_pytorch_model(pt_model, params)

    # Post process the experts

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    pt_model.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--flax_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained SwitchTransformers model. \nThis specifies the"
            " model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_flax_checkpoint_to_pytorch(args.flax_checkpoint_path, args.config_file, args.pytorch_dump_path)
