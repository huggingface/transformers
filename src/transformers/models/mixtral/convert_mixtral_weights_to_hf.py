# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
import re
from glob import glob

import torch

from transformers import AutoTokenizer, MixtralConfig, MixtralForCausalLM


KEYS_TO_MODIFY_MAPPING = {
    "tok_embeddings": "model.embed_tokens",
    "layers": "model.layers",
    "output": "lm_head",
    "wq": "q_proj",
    "wk": "k_proj",
    "wo": "o_proj",
    "wv": "v_proj",
    ".attention.": ".self_attn.",
    ".attention_norm.": ".input_layernorm.",
    ".ffn_norm.": ".post_attention_layernorm.",
}

KEYS_TO_MODIFY_EXACT_MATCH = {"norm.weight": "model.norm.weight"}

torch.set_default_dtype(torch.bfloat16)


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        for key_to_modify, new_key in KEYS_TO_MODIFY_EXACT_MATCH.items():
            if key_to_modify == key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value
    return new_state_dict


def load_and_save_weights(weights_path, save_model_path):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    pt_files = glob(weights_path + "/*.pt")
    config = MixtralConfig(num_hidden_layers=32, hidden_size=1024 * 4, intermediate_size=3584 * 4)

    with torch.device("meta"):
        model = MixtralForCausalLM(config)

    mismatched_weights = []
    num_experts = 8
    for pt_file in pt_files:
        new_state_dict = {}

        partial_state_dict = torch.load(pt_file, map_location="cpu")
        partial_state_dict = convert_state_dict_to_hf(partial_state_dict)

        # Clone parameters to avoid a bug with safetensors
        new_state_dict = {k: v.clone() for k, v in partial_state_dict.items()}
        for k, v in partial_state_dict.items():
            moe_block_match = re.match(r".*.(block_sparse_moe\.w(\d)).*", k)
            if moe_block_match:
                experts = [
                    v[config.intermediate_size * expert_idx : config.intermediate_size * (expert_idx + 1), :]
                    .contiguous()
                    .clone()
                    for expert_idx in range(num_experts)
                ]
                for idx, expert_block in enumerate(experts):
                    expert_key = k.replace("block_sparse_moe.", f"block_sparse_moe.experts.{idx}.")
                    if int(moe_block_match.group(2)) != 2:
                        new_state_dict[expert_key + ".weight"] = expert_block.contiguous().clone()
                    else:
                        new_state_dict[expert_key + ".weight"] = expert_block.T.clone().contiguous()
            else:
                new_state_dict[k] = v.contiguous().clone()

        errors = model.load_state_dict(new_state_dict, strict=False, assign=True)
        mismatched_weights.append(errors)

    for n, p in model.named_parameters():
        assert p.device.type != "meta", f"{n} has not been loaded properly"

    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of Mistral weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    args = parser.parse_args()
    load_and_save_weights(args.input_dir, args.output_dir)
