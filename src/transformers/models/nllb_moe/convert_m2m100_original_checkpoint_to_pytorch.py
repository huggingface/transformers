# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
import json
import argparse
import os
import torch
from torch import nn

from transformers import NllbMoeConfig, NllbMoeForConditionalGeneration
from transformers.models.switch_transformers.convert_switch_transformers_original_flax_checkpoint_to_pytorch import (
    rename_keys,
)
from transformers.modeling_utils import dtype_byte_size
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME

# 'encoder.layers.7.moe_layer.experts.0.fc2.bias', 'encoder.layers.11.moe_layer.experts.0.fc1.weight',

def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "decoder.output_projection.weight",
        "_float_tensor",
        "encoder.embed_positions._float_tensor",
        "decoder.embed_positions._float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

def rename_keys(state_dict):
    # 'encoder.layers.7.moe_layer.experts.0.fc2.bias' ->'encoder.layers.7.ffn.mlp.experts.0.fc2.bias'
    # 'encoder.layers.7.fc2.bias' ->  'encoder.layers.7.ffn.mlp.fc2.bias'
    
    pass

def shard_on_the_fly(switch_checkpoint_path, dump_path, max_shard_size, num_experts, dtype, weights_name: str = WEIGHTS_NAME):
    sharded_state_dicts = []
    current_block = {}
    total_size = 0

    os.makedirs(dump_path, exist_ok=True)

    for expert in range(num_experts):
        expert_path = switch_checkpoint_path + f"-rank-{expert}.pt"
        expert_state = torch.load(expert_path)["model"].keys()
        remove_ignore_keys_(expert_state)
        expert_state = rename_keys(expert_state)
        current_block.update(expert_state)
        save_path = os.path.join(
            dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin")
        )
        torch.save(current_block, save_path)
        sharded_state_dicts.append(current_block.keys())
        total_size += sum([value.numel() for key, value in current_block.items()]) * dtype_byte_size(raw_weights.dtype)

    # Add the last block
    save_path = os.path.join(dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin"))
    shared_weights = torch.load(switch_checkpoint_path.replace(".pt","-shared.pt"))["model"].keys()
    remove_ignore_keys_(shared_weights)
    
    # state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    # lm_head to initialise too

    
    torch.save(shared_weights, save_path)
    sharded_state_dicts.append(shared_weights.keys())

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(
            ".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin"
        )  # len(sharded_state_dicts):05d}
        temp_filename = os.path.join(dump_path, weights_name.replace(".bin", f"-{idx+1:05d}-of-???.bin"))
        os.rename(temp_filename, os.path.join(dump_path, shard_file))
        shards[shard_file] = shard
        for key in shard:
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}

    with open(os.path.join(dump_path, WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)

    return metadata, index





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--nllb_moe_checkpoint_path",
        default="/home/arthur_huggingface_co/fairseq/weights/checkpoints/model_moe_54b/checkpoint_2_300000",
        type=str,
        required=False,
        help="Path to a directory containing a folder per layer. Follows the original Google format.",
    )
    parser.add_argument("--dtype", default="bfloat16", type=str, required=False, help="dtype of the saved model")
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/mnt/disks/disk_switch/original_checkpoints/switch-xxl-128-converted",
        type=str,
        required=False,
        help="Path to the output pytorch model.",
    )
    args = parser.parse_args()
    shard_on_the_fly(
        args.switch_t5x_checkpoint_path,
        args.pytorch_dump_folder_path,
        args.max_shard_size,
        args.dtype,
    )
    
    model = ''
    vocab_size = model["encoder.embed_tokens.weight"].shape[0]

    config = NllbMoeConfig.from_pretrained('',
        num_sparse_encoder_layers=1,
        num_sparse_decoder_layers=1,
        
    )

    model.save_pretrained(args.pytorch_dump_folder_path)
    

