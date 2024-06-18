# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import argparse
import json
import os

import torch
from torch import nn

from transformers import NllbMoeConfig, NllbMoeModel
from transformers.modeling_utils import dtype_byte_size
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME


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


def rename_fairseq_keys(state_dict, expert_idx=None):
    new_dict = {}
    for old_key in state_dict.keys():
        key = old_key
        if "moe_layer.experts." in key:
            if expert_idx is not None:
                key = key.replace("moe_layer.experts.0", f"ffn.experts.expert_{expert_idx}")
            else:
                key = key.replace("moe_layer.experts.", "ffn.experts.expert_")
        if "gate" in key:
            key = key.replace(".moe_layer.gate.wg", ".ffn.router.classifier")
        if "fc2" and "experts" not in key:
            key = key.replace(".fc2.", ".ffn.fc2.")
        if "fc1" and "experts" not in key:
            key = key.replace(".fc1.", ".ffn.fc1.")
        if ".encoder_attn." in key:
            key = key.replace(".encoder_attn.", ".cross_attention.")
        if "encoder_attn_layer_norm" in key:
            key = key.replace("encoder_attn_layer_norm", "cross_attention_layer_norm")
        if "final_layer_norm" in key:
            key = key.replace("final_layer_norm", "ff_layer_norm")
        new_dict[key] = state_dict[old_key]
    return new_dict


def shard_on_the_fly(switch_checkpoint_path, dump_path, num_experts, dtype, weights_name: str = WEIGHTS_NAME):
    sharded_state_dicts = []
    total_size = 0
    os.makedirs(dump_path, exist_ok=True)

    for expert in range(num_experts):
        expert_path = switch_checkpoint_path + f"-rank-{expert}.pt"
        if os.path.isfile(expert_path):
            expert_state = torch.load(expert_path)["model"]
            remove_ignore_keys_(expert_state)
            expert_state = rename_fairseq_keys(expert_state, expert)
            save_path = os.path.join(
                dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin")
            )
            torch.save(expert_state, save_path)
            sharded_state_dicts.append(expert_state.keys())
            total_size += sum([value.numel() for key, value in expert_state.items()]) * dtype_byte_size(
                expert_state[list(expert_state)[0]].dtype
            )

    # Add the last block
    save_path = os.path.join(dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin"))
    shared_weights = torch.load(switch_checkpoint_path + "-shared.pt")["model"]
    remove_ignore_keys_(shared_weights)
    shared_weights = rename_fairseq_keys(shared_weights, None)
    shared_weights["shared.weight"] = shared_weights["decoder.embed_tokens.weight"]
    sharded_state_dicts.append(shared_weights.keys())

    # If we only have the shared weights (dummy model/experts saved on the same file)
    if len(sharded_state_dicts) == 1:
        save_path = os.path.join(dump_path, weights_name)
        torch.save(shared_weights, save_path)
        return {weights_name: sharded_state_dicts[0]}, None
    else:
        torch.save(shared_weights, save_path)
    # Otherwise, let's build the index
    weight_map = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        temp_filename = os.path.join(dump_path, weights_name.replace(".bin", f"-{idx+1:05d}-of-???.bin"))
        os.rename(temp_filename, os.path.join(dump_path, shard_file))
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
    parser.add_argument("--dtype", default="float32", type=str, required=False, help="dtype of the saved model")
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/home/arthur_huggingface_co/fairseq/weights/checkpoints/hf-converted-moe-54b",
        type=str,
        required=False,
        help="Path to the output pytorch model.",
    )
    args = parser.parse_args()
    metadata, index = shard_on_the_fly(
        args.nllb_moe_checkpoint_path,
        args.pytorch_dump_folder_path,
        128,
        args.dtype,
    )

    config = NllbMoeConfig.from_pretrained(
        "facebook/nllb-200-3.3B", encoder_sparse_step=4, decoder_sparse_step=4, num_experts=128
    )
    config.save_pretrained(args.pytorch_dump_folder_path)
    model = NllbMoeModel.from_pretrained(args.pytorch_dump_folder_path)
    print("Done")
    model.save_pretrained(args.pytorch_dump_folder_path)
