import argparse
import json
import os
import re

import torch
from safetensors.torch import load_file

from transformers import DogeConfig, DogeForCausalLM


# fmt: off
# `None` means we drop the key
STATE_DICT_MAPPING = {
    # CausalLM keys
    r"^lm_head.weight": r"lm_head.weight",

    # Model keys
    r"^model.word_embed.weight": r"model.embed_tokens.weight",
    r"^model.rotary_emb.rotary_emb": r"model.rotary_emb.rotary_emb",
    r"^model.final_layernorm.weight": r"model.norm.weight",

    # Layers keys
    r"^model.layers.(\d+).pre_layernorm.weight": r"model.layers.\1.input_layernorm.weight",
    r"^model.layers.(\d+).pre_residual.weight": r"model.layers.\1.input_residual",
    r"^model.layers.(\d+).post_layernorm.weight": r"model.layers.\1.post_attention_layernorm.weight",
    r"^model.layers.(\d+).post_residual.weight": r"model.layers.\1.post_attention_residual",

    # Attention keys
    r"^model.layers.(\d+).self_attn.q_proj.weight": r"model.layers.\1.self_attn.q_proj.weight",
    r"^model.layers.(\d+).self_attn.k_proj.weight": r"model.layers.\1.self_attn.k_proj.weight",
    r"^model.layers.(\d+).self_attn.v_proj.weight": r"model.layers.\1.self_attn.v_proj.weight",
    r"^model.layers.(\d+).self_attn.A": r"model.layers.\1.self_attn.A",
    r"^model.layers.(\d+).self_attn.dt_proj.weight": r"model.layers.\1.self_attn.dt_proj.weight",
    r"^model.layers.(\d+).self_attn.o_proj.weight": r"model.layers.\1.self_attn.o_proj.weight",

    # Feedforward keys
    r"^model.layers.(\d+).feed_forward.gate_proj.weight": r"model.layers.\1.mlp.gate_proj.weight",
    r"^model.layers.(\d+).feed_forward.up_proj.weight": r"model.layers.\1.mlp.up_proj.weight",
    r"^model.layers.(\d+).feed_forward.down_proj.weight": r"model.layers.\1.mlp.down_proj.weight",
    r"^model.layers.(\d+).feed_forward.router_gate.weight": r"model.layers.\1.mlp.router_gate.weight",
    r"^model.layers.(\d+).feed_forward.router_gate.bias": None,
    r"^model.layers.(\d+).feed_forward.down_embed.weight": r"model.layers.\1.mlp.down_embed.weight",
    r"^model.layers.(\d+).feed_forward.up_embed.weight": r"model.layers.\1.mlp.up_embed.weight",
}
# fmt: on


def load_weights(input_dir: str):
    safetensor_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if x.endswith(".safetensors")]

    all_weights = {}

    if safetensor_files:
        if len(safetensor_files) == 1:
            tensors = load_file(safetensor_files[0])
            all_weights.update(tensors)
            return all_weights
        safetensor_files = sorted(safetensor_files, key=lambda x: int(x.rsplit("-", 3)[1]))
        for file in safetensor_files:
            tensors = load_file(file)
            all_weights.update(tensors)
        return all_weights

    else:
        raise ValueError("No .safetensors or .bin files found in the specified directory.")


def map_old_key_to_new(old_key):
    for pattern, replacement in STATE_DICT_MAPPING.items():
        if replacement is None:
            if re.fullmatch(pattern, old_key):
                return None
        else:
            new_key, n_replace = re.subn(pattern, replacement, old_key)
            # Early exit of the loop
            if n_replace > 0:
                return new_key

    raise ValueError(f"Key: {old_key} could not be mapped (check the mapping).")


def convert_state_dict(original_state_dict: dict, config: DogeConfig):
    new_dict = {}

    for old_key, value in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)
        if new_key is None:
            continue
        new_dict[new_key] = value
    return new_dict


def convert_doge_model(input_dir, output_dir):
    # Load and convert config
    with open(os.path.join(input_dir, "config.json")) as f:
        config = json.load(f)
    config = DogeConfig(**config)
    config.save_pretrained(output_dir)

    # Load and convert weights
    original_state_dict = load_weights(input_dir)
    new_dict = convert_state_dict(original_state_dict, config)
    with torch.device("meta"):
        model = DogeForCausalLM(config)
    if config.tie_word_embeddings:
        new_dict["lm_head.weight"] = new_dict["model.embed_tokens.weight"]
    model.load_state_dict(new_dict, strict=True, assign=True)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="Location of the local folder copied from the Hub.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Location to write HF model.",
    )

    args = parser.parse_args()
    convert_doge_model(args.input_dir, args.output_dir)
