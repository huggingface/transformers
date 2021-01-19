# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert Blenderbot checkpoint."""

import argparse

import torch

from transformers import BartConfig, BartForConditionalGeneration
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

PATTERNS = [
    ["attention", "attn"],
    ["encoder_attention", "encoder_attn"],
    ["q_lin", "q_proj"],
    ["k_lin", "k_proj"],
    ["v_lin", "v_proj"],
    ["out_lin", "out_proj"],
    ["norm_embeddings", "layernorm_embedding"],
    ["position_embeddings", "embed_positions"],
    ["embeddings", "embed_tokens"],
    ["ffn.lin", "fc"],
]


def rename_state_dict_key(k):
    if k == "embeddings.weight":
        return "shared.weight"

    for parlai_name, hf_name in PATTERNS:
        k = k.replace(parlai_name, hf_name)

    if k.startswith("encoder"):
        k = k.replace(".attn", ".self_attn")
        k = k.replace("norm1", "self_attn_layer_norm")
        k = k.replace("norm2", "final_layer_norm")
    elif k.startswith("decoder"):
        k = k.replace("norm1", "self_attn_layer_norm")
        k = k.replace("norm2", "encoder_attn_layer_norm")
        k = k.replace("norm3", "final_layer_norm")
    return k


def rename_layernorm_keys(sd):
    keys = [
        "model.encoder.layernorm_embedding.weight",
        "model.encoder.layernorm_embedding.bias",
        "model.decoder.layernorm_embedding.weight",
        "model.decoder.layernorm_embedding.bias",
    ]
    for k in keys:
        v = sd.pop(k)
        new_k = k.replace("layernorm_embedding", "layer_norm")
        assert new_k not in sd
        sd[new_k] = v


IGNORE_KEYS = ["START"]


@torch.no_grad()
def convert_parlai_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_json_path):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    model = torch.load(checkpoint_path, map_location="cpu")
    sd = model["model"]
    cfg = BartConfig.from_json_file(config_json_path)
    m = BartForConditionalGeneration(cfg)
    valid_keys = m.model.state_dict().keys()
    failures = []
    mapping = {}
    for k, v in sd.items():
        if k in IGNORE_KEYS:
            continue

        new_k = rename_state_dict_key(k)
        if new_k not in valid_keys:
            failures.append([k, new_k])
        else:
            mapping[new_k] = v
    if cfg.normalize_before:  # Blenderbot-3B checkpoints. Rename layernorm_embedding -> layer_norm
        rename_layernorm_keys(sd)
    m.model.load_state_dict(mapping, strict=True)
    m.half()
    m.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--src_path", type=str, help="like blenderbot-model.bin")
    parser.add_argument("--save_dir", default="hf_blenderbot", type=str, help="Where to save converted model.")
    parser.add_argument(
        "--hf_config_json", default="blenderbot-3b-config.json", type=str, help="Path to config to use"
    )
    args = parser.parse_args()
    convert_parlai_checkpoint(args.src_path, args.save_dir, args.hf_config_json)
