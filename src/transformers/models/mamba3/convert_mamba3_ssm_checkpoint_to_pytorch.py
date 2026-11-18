# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Converts a `mamba_ssm` Mamba-3 checkpoint (e.g. `state-spaces/mamba3-siso-187m`) to transformers."""

import argparse
import json

import torch
from huggingface_hub import hf_hub_download

from transformers import AutoTokenizer, Mamba3Config, Mamba3ForCausalLM


def convert_config(ssm_config: dict) -> Mamba3Config:
    ssm_cfg = ssm_config["ssm_cfg"]
    if ssm_cfg.get("is_mimo") or ssm_cfg.get("is_outproj_norm") or ssm_config.get("attn_layer_idx"):
        raise ValueError("Only SISO Mamba-3 checkpoints without out-proj norm or attention layers are supported.")
    vocab_size = ssm_config["vocab_size"]
    pad_multiple = ssm_config.get("pad_vocab_size_multiple", 1)
    vocab_size += (pad_multiple - vocab_size % pad_multiple) % pad_multiple
    hidden_size, expand, head_dim = ssm_config["d_model"], ssm_cfg["expand"], ssm_cfg["headdim"]
    return Mamba3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=ssm_config["d_intermediate"],
        num_hidden_layers=ssm_config["n_layer"],
        num_heads=hidden_size * expand // head_dim,
        head_dim=head_dim,
        expand=expand,
        state_size=ssm_cfg["d_state"],
        n_groups=ssm_cfg["ngroups"],
        rope_fraction=ssm_cfg["rope_fraction"],
        a_floor=ssm_cfg["A_floor"],
        chunk_size=ssm_cfg["chunk_size"],
        time_step_min=ssm_cfg["dt_min"],
        time_step_max=ssm_cfg["dt_max"],
        time_step_floor=ssm_cfg["dt_init_floor"],
        residual_in_fp32=ssm_config["residual_in_fp32"],
        tie_word_embeddings=ssm_config["tie_embeddings"],
    )


def convert_state_dict(state_dict: dict) -> dict:
    converted = {}
    for key, value in state_dict.items():
        key = key.replace("backbone.embedding.", "backbone.embeddings.").replace(".norm2.", ".mlp_norm.")
        if key.endswith(("mixer.B_bias", "mixer.C_bias")):
            value = value.squeeze(1)  # (num_heads, mimo_rank=1, state_size)
        if key.endswith("mlp.fc1.weight"):
            # `mamba_ssm.GatedMLP` computes `y * silu(gate)` with `y, gate = fc1(x).chunk(2)`
            up, gate = value.chunk(2, dim=0)
            converted[key.replace("fc1", "up_proj")] = up
            converted[key.replace("fc1", "gate_proj")] = gate
            continue
        converted[key.replace("mlp.fc2.", "mlp.down_proj.")] = value
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="state-spaces/mamba3-siso-187m")
    parser.add_argument("--tokenizer_id", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--push_to_hub", default=None, help="Optional Hub repo id to push the converted model to.")
    args = parser.parse_args()

    with open(hf_hub_download(args.repo_id, "config.json"), encoding="utf-8") as f:
        config = convert_config(json.load(f))
    state_dict = torch.load(hf_hub_download(args.repo_id, "pytorch_model.bin"), map_location="cpu", weights_only=True)

    model = Mamba3ForCausalLM(config)
    missing, unexpected = model.load_state_dict(convert_state_dict(state_dict), strict=False)
    missing = [key for key in missing if not (config.tie_word_embeddings and key == "lm_head.weight")]
    if missing or unexpected:
        raise ValueError(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub, private=True)
        tokenizer.push_to_hub(args.push_to_hub, private=True)


if __name__ == "__main__":
    main()
