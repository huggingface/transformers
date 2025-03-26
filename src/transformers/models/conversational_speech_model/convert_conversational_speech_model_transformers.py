# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import os
import re
import json
import torch

from transformers import (
    ConversationalSpeechModelConfig,
    ConversationalSpeechModelForCausalLM,
)
from transformers.utils.hub import get_file_from_repo


# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"backbone\.layers\.(\d+)":                r"backbone_model.layers.\1",
    r"decoder\.layers\.(\d+)":            r"depth_decoder.model.layers.\1",

    r"attn":                                                  r"self_attn",
    r"output_proj":                                              r"o_proj",
    r"w1":                                                    r"gate_proj",
    r"w2":                                                    r"down_proj",
    r"w3":                                                      r"up_proj",

    r"text_embeddings":   r"backbone_model.embed_tokens.embed_text_tokens",
    r"audio_embeddings": r"backbone_model.embed_tokens.embed_audio_tokens",

    r"codebook0_head":                                          r"lm_head",
    r"audio_head":                  r"depth_decoder.codebooks_head.weight",
    r"projection":          r"depth_decoder.model.inputs_embeds_projector",

    r"sa_norm.scale":                            r"input_layernorm.weight",
    r"mlp_norm.scale":                  r"post_attention_layernorm.weight",
    r"decoder.norm.scale":              r"depth_decoder.model.norm.weight",
    r"backbone.norm.scale":                  r"backbone_model.norm.weight",
}
# fmt: on

def permute_for_rope(input_tensor, n_heads, dim1, dim2):
    """
    When you go from the complex ROPE formulation to sin and cos one, you need
    to permute the query and key weights (to avoid doing it on the fly)
    """
    input_tensor = input_tensor.reshape(dim1, dim2)
    input_tensor = input_tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    input_tensor = input_tensor.transpose(1, 2).reshape(dim1, dim2)
    return input_tensor


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def write_model(
    input_path_or_repo,
    model_name,
    output_dir,
    safe_serialization=True,
):
    print("Converting the model.")
    os.makedirs(output_dir, exist_ok=True)

    config = ConversationalSpeechModelConfig()
    params = {
        "backbone": {
            "num_attention_heads": config.backbone_config.num_attention_heads,
            "num_key_value_heads": config.backbone_config.num_key_value_heads,
            "dim_per_head": config.backbone_config.head_dim,
            "key_value_dim": config.backbone_config.head_dim * config.backbone_config.num_key_value_heads,
            "dim": config.backbone_config.hidden_size,
        },
        "depth_decoder": {
            "num_attention_heads": config.depth_decoder_config.num_attention_heads,
            "num_key_value_heads": config.depth_decoder_config.num_key_value_heads,
            "dim_per_head": config.depth_decoder_config.head_dim,
            "key_value_dim": config.depth_decoder_config.head_dim * config.depth_decoder_config.num_key_value_heads,
            "dim": config.depth_decoder_config.hidden_size,
        },
    }

    model_path = get_file_from_repo(
        input_path_or_repo,
        model_name,
    )
    print(f"Fetching all parameters from the checkpoint at {model_path}...")
    loaded = torch.load(model_path, map_location="cpu")

    print("Converting model...")
    state_dict = {}

    # -----------------------
    # convert parameter names
    # -----------------------

    for key, value in loaded.items():
        new_key = convert_key(key, ORIGINAL_TO_CONVERTED_KEY_MAPPING)
        current_parameter = value
        
        # Post-process the current_parameter.
        if re.search("(k|q)_proj.weight", new_key):
            params_keys = "backbone" if "backbone" in new_key else "depth_decoder"
            if "q_proj" in new_key:
                num_heads = params[params_keys]["num_attention_heads"]
                dim_per_head = params[params_keys]["dim_per_head"]
                param_dim = params[params_keys]["dim"]
                dim = params[params_keys]["dim"]
            else:
                num_heads = params[params_keys]["num_key_value_heads"]
                dim_per_head = params[params_keys]["dim_per_head"]
                param_dim = params[params_keys]["key_value_dim"]
                dim = params[params_keys]["dim"]

            current_parameter = permute_for_rope(value, num_heads, param_dim, dim)
            state_dict[new_key] = current_parameter.reshape(num_heads * dim_per_head, dim)

        state_dict[new_key] = current_parameter

    # add the depth decoder embed audio tokens weights, latter tied to the backbone embed audio tokens weights
    state_dict["depth_decoder.model.embed_tokens.embed_audio_tokens.weight"] = state_dict["backbone_model.embed_tokens.embed_audio_tokens.weight"]

    # -------------------------
    # load the weights and save
    # -------------------------

    model = ConversationalSpeechModelForCausalLM(config)
    model.load_state_dict(state_dict)

    print("Saving the model...")
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    print(f"Model saved at {output_dir}!")


def main():
    parser = argparse.ArgumentParser(description="Convert CSM weights to HuggingFace format")
    parser.add_argument(
        "--input_path_or_repo",
        type=str,
        required=True,
        help="Path or repo containing CSM weights",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model in input_path_or_repo",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", action="store_true", default=True, help="Whether or not to save using `safetensors`."
    )
    args = parser.parse_args()

    write_model(
        args.input_path_or_repo,
        args.model_name,
        output_dir=args.output_dir, 
        safe_serialization=args.safe_serialization,
    )


if __name__ == "__main__":

    write_model(
        "sesame/csm-1b",
        "ckpt.pt",
        output_dir="eustlb/csm-1b",
        safe_serialization=True,
    )
