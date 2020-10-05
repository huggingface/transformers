# coding=utf-8
# Copyright 2020 Google and The HuggingFace Inc. team.
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
from pathlib import Path
from typing import Dict

import tensorflow as tf
import torch
from tqdm import tqdm

from transformers import PegasusConfig, PegasusForConditionalGeneration, PegasusTokenizer
from transformers.configuration_pegasus import DEFAULTS, task_specific_params


PATTERNS = [
    # replace left string with right string to get the relevant state_dict key (identical state dict to bart)
    ["memory_attention", "encoder_attn"],
    ["attention", "attn"],
    ["/", "."],
    [".LayerNorm.gamma", "_layer_norm.weight"],
    [".LayerNorm.beta", "_layer_norm.bias"],
    ["r.layer_", "r.layers."],
    ["output_proj", "out_proj"],
    ["ffn.dense_1.", "fc2."],
    ["ffn.dense.", "fc1."],
    ["ffn_layer_norm", "final_layer_norm"],
    ["kernel", "weight"],
    ["encoder_layer_norm.", "encoder.layer_norm."],
    ["decoder_layer_norm.", "decoder.layer_norm."],
    ["embeddings.weights", "shared.weight"],
]


def rename_state_dict_key(k):

    for pegasus_name, hf_name in PATTERNS:
        k = k.replace(pegasus_name, hf_name)
    return k


# See appendix C of paper for all hyperparams

# TODO(SS): one constant


def convert_pegasus(tf_weights: dict, cfg_updates: dict) -> PegasusForConditionalGeneration:
    cfg_kwargs = DEFAULTS.copy()
    cfg_kwargs.update(cfg_updates)
    cfg = PegasusConfig(**cfg_kwargs)
    torch_model = PegasusForConditionalGeneration(cfg)
    sd = torch_model.model.state_dict()
    mapping = {}
    for k, v in tf_weights.items():
        new_k = rename_state_dict_key(k)
        if new_k not in sd:
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")

        if "dense" in k or "proj" in new_k:
            v = v.T
        mapping[new_k] = torch.tensor(v, dtype=sd[new_k].dtype)
        assert v.shape == sd[new_k].shape, f"{new_k}, {k}, {v.shape}, {sd[new_k].shape}"
    # make sure embedding.padding_idx is respected
    mapping["shared.weight"][cfg.pad_token_id] = torch.zeros_like(mapping["shared.weight"][cfg.pad_token_id + 1])
    mapping["encoder.embed_tokens.weight"] = mapping["shared.weight"]
    mapping["decoder.embed_tokens.weight"] = mapping["shared.weight"]
    empty_biases = {k: torch.zeros_like(v) for k, v in sd.items() if k.endswith("bias") and k not in mapping}
    mapping.update(**empty_biases)
    missing, extra = torch_model.model.load_state_dict(mapping, strict=False)
    unexpected_missing = [
        k for k in missing if k not in ["encoder.embed_positions.weight", "decoder.embed_positions.weight"]
    ]
    assert unexpected_missing == [], f"no matches found for the following torch keys {unexpected_missing}"
    assert extra == [], f"no matches found for the following tf keys {extra}"
    return torch_model


def get_tf_weights_as_numpy(path="./ckpt/aeslc/model.ckpt-32000") -> Dict:
    init_vars = tf.train.list_variables(path)
    tf_weights = {}
    ignore_name = ["Adafactor", "global_step"]
    for name, shape in tqdm(init_vars, desc="converting tf checkpoint to dict"):
        skip_key = any([pat in name for pat in ignore_name])
        if skip_key:
            continue
        array = tf.train.load_variable(path, name)
        tf_weights[name] = array
    return tf_weights


def convert_pegasus_ckpt_to_pytorch(ckpt_path: str, save_dir: str):
    # save tokenizer first
    dataset = Path(ckpt_path).parent.name
    desired_max_model_length = task_specific_params[f"summarization_{dataset}"]["max_position_embeddings"]
    tok = PegasusTokenizer.from_pretrained("sshleifer/pegasus", model_max_length=desired_max_model_length)
    assert tok.model_max_length == desired_max_model_length
    tok.save_pretrained(save_dir)

    # convert model
    tf_weights = get_tf_weights_as_numpy(ckpt_path)
    cfg_updates = task_specific_params[f"summarization_{dataset}"]
    if dataset == "large":
        cfg_updates["task_specific_params"] = task_specific_params
    torch_model = convert_pegasus(tf_weights, cfg_updates)
    torch_model.save_pretrained(save_dir)
    sd = torch_model.state_dict()
    sd.pop("model.decoder.embed_positions.weight")
    sd.pop("model.encoder.embed_positions.weight")
    torch.save(sd, Path(save_dir) / "pytorch_model.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("tf_ckpt_path", type=str, help="passed to tf.train.list_variables")
    parser.add_argument("save_dir", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    if args.save_dir is None:
        dataset = Path(args.tf_ckpt_path).parent.name
        args.save_dir = os.path.join("pegasus", dataset)
    convert_pegasus_ckpt_to_pytorch(args.tf_ckpt_path, args.save_dir)
