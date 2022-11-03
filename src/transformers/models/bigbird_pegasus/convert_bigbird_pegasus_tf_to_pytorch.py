# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
from typing import Dict

import tensorflow as tf
import torch
from tqdm import tqdm

from transformers import BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration


INIT_COMMON = [
    # tf -> hf
    ("/", "."),
    ("layer_", "layers."),
    ("kernel", "weight"),
    ("beta", "bias"),
    ("gamma", "weight"),
    ("pegasus", "model"),
]
END_COMMON = [
    (".output.dense", ".fc2"),
    ("intermediate.LayerNorm", "final_layer_norm"),
    ("intermediate.dense", "fc1"),
]

DECODER_PATTERNS = (
    INIT_COMMON
    + [
        ("attention.self.LayerNorm", "self_attn_layer_norm"),
        ("attention.output.dense", "self_attn.out_proj"),
        ("attention.self", "self_attn"),
        ("attention.encdec.LayerNorm", "encoder_attn_layer_norm"),
        ("attention.encdec_output.dense", "encoder_attn.out_proj"),
        ("attention.encdec", "encoder_attn"),
        ("key", "k_proj"),
        ("value", "v_proj"),
        ("query", "q_proj"),
        ("decoder.LayerNorm", "decoder.layernorm_embedding"),
    ]
    + END_COMMON
)

REMAINING_PATTERNS = (
    INIT_COMMON
    + [
        ("embeddings.word_embeddings", "shared.weight"),
        ("embeddings.position_embeddings", "embed_positions.weight"),
        ("attention.self.LayerNorm", "self_attn_layer_norm"),
        ("attention.output.dense", "self_attn.output"),
        ("attention.self", "self_attn.self"),
        ("encoder.LayerNorm", "encoder.layernorm_embedding"),
    ]
    + END_COMMON
)

KEYS_TO_IGNORE = [
    "encdec/key/bias",
    "encdec/query/bias",
    "encdec/value/bias",
    "self/key/bias",
    "self/query/bias",
    "self/value/bias",
    "encdec_output/dense/bias",
    "attention/output/dense/bias",
]


def rename_state_dict_key(k, patterns):
    for tf_name, hf_name in patterns:
        k = k.replace(tf_name, hf_name)
    return k


def convert_bigbird_pegasus(tf_weights: dict, config_update: dict) -> BigBirdPegasusForConditionalGeneration:
    cfg = BigBirdPegasusConfig(**config_update)
    torch_model = BigBirdPegasusForConditionalGeneration(cfg)
    state_dict = torch_model.state_dict()
    mapping = {}

    # separating decoder weights
    decoder_weights = {k: tf_weights[k] for k in tf_weights if k.startswith("pegasus/decoder")}
    remaining_weights = {k: tf_weights[k] for k in tf_weights if not k.startswith("pegasus/decoder")}

    for k, v in tqdm(decoder_weights.items(), "tf -> hf conversion"):
        conditions = [k.endswith(ending) for ending in KEYS_TO_IGNORE]
        if any(conditions):
            continue
        patterns = DECODER_PATTERNS
        new_k = rename_state_dict_key(k, patterns)
        if new_k not in state_dict:
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")
        if any([True if i in k else False for i in ["dense", "query", "key", "value"]]):
            v = v.T
        mapping[new_k] = torch.from_numpy(v)
        assert v.shape == state_dict[new_k].shape, f"{new_k}, {k}, {v.shape}, {state_dict[new_k].shape}"

    for k, v in tqdm(remaining_weights.items(), "tf -> hf conversion"):
        conditions = [k.endswith(ending) for ending in KEYS_TO_IGNORE]
        if any(conditions):
            continue
        patterns = REMAINING_PATTERNS
        new_k = rename_state_dict_key(k, patterns)
        if new_k not in state_dict and k != "pegasus/embeddings/position_embeddings":
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")
        if any([True if i in k else False for i in ["dense", "query", "key", "value"]]):
            v = v.T
        mapping[new_k] = torch.from_numpy(v)
        if k != "pegasus/embeddings/position_embeddings":
            assert v.shape == state_dict[new_k].shape, f"{new_k}, {k}, {v.shape}, {state_dict[new_k].shape}"

    mapping["model.encoder.embed_positions.weight"] = mapping["model.embed_positions.weight"]
    mapping["model.decoder.embed_positions.weight"] = mapping.pop("model.embed_positions.weight")
    missing, extra = torch_model.load_state_dict(mapping, strict=False)
    unexpected_missing = [
        k
        for k in missing
        if k
        not in [
            "final_logits_bias",
            "model.encoder.embed_tokens.weight",
            "model.decoder.embed_tokens.weight",
            "lm_head.weight",
        ]
    ]
    assert unexpected_missing == [], f"no matches found for the following torch keys {unexpected_missing}"
    assert extra == [], f"no matches found for the following tf keys {extra}"
    return torch_model


def get_tf_weights_as_numpy(path) -> Dict:
    init_vars = tf.train.list_variables(path)
    tf_weights = {}
    ignore_name = ["global_step"]
    for name, shape in tqdm(init_vars, desc="converting tf checkpoint to dict"):
        skip_key = any([pat in name for pat in ignore_name])
        if skip_key:
            continue
        array = tf.train.load_variable(path, name)
        tf_weights[name] = array
    return tf_weights


def convert_bigbird_pegasus_ckpt_to_pytorch(ckpt_path: str, save_dir: str, config_update: dict):
    tf_weights = get_tf_weights_as_numpy(ckpt_path)
    torch_model = convert_bigbird_pegasus(tf_weights, config_update)
    torch_model.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_ckpt_path", type=str, help="passed to tf.train.list_variables")
    parser.add_argument("--save_dir", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    config_update = {}
    convert_bigbird_pegasus_ckpt_to_pytorch(args.tf_ckpt_path, args.save_dir, config_update=config_update)
