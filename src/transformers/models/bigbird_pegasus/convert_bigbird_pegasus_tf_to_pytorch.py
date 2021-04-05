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

from transformers import BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration, BigBirdPegasusTokenizer


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
    # ffn
    (".output.dense", ".fc2"),
    ("intermediate.LayerNorm", "final_layer_norm"),
    ("intermediate.dense", "fc1"),
]

DECODER_PATTERNS = INIT_COMMON + [
    # self-attn
    ("attention.self.LayerNorm", "self_attn_layer_norm"),
    ("attention.output.dense", "self_attn.out_proj"),
    ("attention.self", "self_attn"),
    # cross-attn
    ("attention.encdec.LayerNorm", "encoder_attn_layer_norm"),
    ("attention.encdec_output.dense", "encoder_attn.out_proj"),
    ("attention.encdec", "encoder_attn"),
    ("key", "k_proj"),
    ("value", "v_proj"),
    ("query", "q_proj"),
    ("decoder.LayerNorm", "decoder.layernorm_embedding"),
] + END_COMMON

REMAINING_PATTERNS = INIT_COMMON + [
    # embedding
    ("embeddings.word_embeddings", "shared.weight"),
    ("embeddings.position_embeddings", "embed_positions.weight"),
    # self-attn
    ("attention.self.LayerNorm", "self_attn_layer_norm"),
    ("attention.output.dense", "self_attn.output"),
    ("attention.self", "self_attn.self"),
    ("encoder.LayerNorm", "encoder.layernorm_embedding"),
] + END_COMMON


def rename_state_dict_key(k, patterns):
    for tf_name, hf_name in patterns:
        k = k.replace(tf_name, hf_name)
    return k


def convert_bigbird_pegasus(tf_weights: dict, config_update: dict) -> BigBirdPegasusForConditionalGeneration:

    cfg = BigBirdPegasusConfig(**config_update)
    torch_model = BigBirdPegasusForConditionalGeneration(cfg)
    sd = torch_model.state_dict()
    mapping = {}

    # seperating decoder weights
    decoder_weights = {k: tf_weights[k] for k in tf_weights if k.startswith("pegasus/decoder")}
    remaining_weights = {k: tf_weights[k] for k in tf_weights if not k.startswith("pegasus/decoder")}

    for k, v in tqdm(decoder_weights.items(), "tf -> hf conversion"):
        patterns = DECODER_PATTERNS
        new_k = rename_state_dict_key(k, patterns)
        # print(k, " -> ", new_k)
        if new_k not in sd:
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")
        if "dense" in k:
            v = v.T
        mapping[new_k] = torch.from_numpy(v)
        assert v.shape == sd[new_k].shape, f"{new_k}, {k}, {v.shape}, {sd[new_k].shape}"

    for k, v in tqdm(remaining_weights.items(), "tf -> hf conversion"):
        patterns = REMAINING_PATTERNS
        new_k = rename_state_dict_key(k, patterns)
        # print(k, " -> ", new_k)
        if new_k not in sd and k != "pegasus/embeddings/position_embeddings":
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")
        if "dense" in k:
            v = v.T
        mapping[new_k] = torch.from_numpy(v)
        if k != "pegasus/embeddings/position_embeddings":
            assert v.shape == sd[new_k].shape, f"{new_k}, {k}, {v.shape}, {sd[new_k].shape}"

    # make sure embedding.padding_idx is respected
    # mapping["shared.weight"][cfg.pad_token_id] = torch.zeros_like(mapping["shared.weight"][cfg.pad_token_id + 1])
    mapping["model.encoder.embed_positions.weight"] = mapping["model.embed_positions.weight"]
    mapping["model.decoder.embed_positions.weight"] = mapping.pop("model.embed_positions.weight")
    # empty_biases = {k: torch.zeros_like(v) for k, v in sd.items() if k.endswith("bias") and k not in mapping}
    # mapping.update(**empty_biases)
    missing, extra = torch_model.load_state_dict(mapping, strict=False)
    unexpected_missing = [
        k for k in missing if k not in ['final_logits_bias', 'model.encoder.embed_tokens.weight', 'model.decoder.embed_tokens.weight', 'lm_head.weight']
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
    # save tokenizer first
    # dataset = Path(ckpt_path).parent.name
    # desired_max_model_length = task_specific_params[f"summarization_{dataset}"]["max_position_embeddings"]
    # tok = BigBirdPegasusTokenizer.from_pretrained("sshleifer/pegasus", model_max_length=desired_max_model_length)
    # assert tok.model_max_length == desired_max_model_length
    # tok.save_pretrained(save_dir)

    # convert model
    tf_weights = get_tf_weights_as_numpy(ckpt_path)
    torch_model = convert_bigbird_pegasus(tf_weights, config_update)
    torch_model.save_pretrained(save_dir)
    # sd = torch_model.state_dict()
    # sd.pop("model.decoder.embed_positions.weight")
    # sd.pop("model.encoder.embed_positions.weight")
    # torch.save(sd, os.path.join(save_dir, "pytorch_model.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--tf_ckpt_path", type=str, help="passed to tf.train.list_variables")
    parser.add_argument("--save_dir", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    config_update = {}
    convert_bigbird_pegasus_ckpt_to_pytorch(args.tf_ckpt_path, args.save_dir, config_update=config_update)


# TODO:
# _ = [print(a[0], a[1]) for a in tf.train.list_variables("src/tf_ckpt/bigbird-pegasus-large-arxiv/model.ckpt-0")]
# python3 src/transformers/models/bigbird_pegasus/convert_pegasus_tf_to_pytorch.py --tf_ckpt_path src/tf_ckpt/bigbird-pegasus-large-arxiv/model.ckpt-0 --save_dir src/ckpt/bigbird-pegasus-large-arxiv
