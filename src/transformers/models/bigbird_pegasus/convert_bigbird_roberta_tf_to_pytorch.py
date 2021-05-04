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

from transformers import BigBirdConfig, EncoderDecoderConfig, EncoderDecoderModel


PATTERNS = [
    # tf -> hf
    ("/", "."),
    ("layer_", "layer."),
    ("kernel", "weight"),
    ("beta", "bias"),
    ("gamma", "weight"),
    ("attention.encdec.", "crossattention.self."),
    ("attention.encdec_", "crossattention."),
    ("encoder.LayerNorm", "embeddings.LayerNorm"),
    ("word_embeddings", "word_embeddings.weight"),
    ("position_embeddings", "position_embeddings.weight"),
]


def rename_state_dict_key(k):
    for tf_name, hf_name in PATTERNS:
        k = k.replace(tf_name, hf_name)
    return k


def convert_bigbird_roberta(tf_weights: dict) -> EncoderDecoderModel:

    encoder_config = BigBirdConfig(type_vocab_size=1)
    decoder_config = BigBirdConfig(attention_type="original_full", type_vocab_size=1)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    torch_model = EncoderDecoderModel(config=config)
    sd = torch_model.state_dict()
    mapping = {}

    # decoder
    for k, v in tqdm(tf_weights.items(), "tf -> hf conversion"):
        new_k = rename_state_dict_key(k)
        new_k = "decoder." + new_k
        # print(k, " -> ", new_k)
        if new_k not in sd:
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")
        if any([True if i in k else False for i in ["dense", "query", "key", "value"]]):
            v = v.T
        mapping[new_k] = torch.from_numpy(v)
        assert v.shape == sd[new_k].shape, f"{new_k}, {k}, {v.shape}, {sd[new_k].shape}"

    # encoder
    for k, v in tqdm(tf_weights.items(), "tf -> hf conversion"):
        new_k = rename_state_dict_key(k)
        new_k = new_k.replace("bert", "encoder")
        if "crossattention" in new_k:
            continue
        # print(k, " -> ", new_k)
        if new_k not in sd:
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")
        if any([True if i in k else False for i in ["dense", "query", "key", "value"]]):
            v = v.T
        mapping[new_k] = torch.from_numpy(v)
        assert v.shape == sd[new_k].shape, f"{new_k}, {k}, {v.shape}, {sd[new_k].shape}"

    # make sure embedding.padding_idx is respected
    # mapping["shared.weight"][cfg.pad_token_id] = torch.zeros_like(mapping["shared.weight"][cfg.pad_token_id + 1])
    mapping["encoder.embeddings.token_type_embeddings.weight"] = torch.zeros(1, 768, dtype=torch.long)
    mapping["decoder.bert.embeddings.token_type_embeddings.weight"] = torch.zeros(1, 768, dtype=torch.long)
    missing, extra = torch_model.load_state_dict(mapping, strict=False)
    unexpected_missing = [
        k
        for k in missing
        if k
        not in [
            "encoder.embeddings.position_ids",
            "encoder.pooler.weight",
            "encoder.pooler.bias",
            "decoder.bert.embeddings.position_ids",
            "decoder.bert.pooler.weight",
            "decoder.bert.pooler.bias",
            "decoder.cls.predictions.bias",
            "decoder.cls.predictions.transform.dense.weight",
            "decoder.cls.predictions.transform.dense.bias",
            "decoder.cls.predictions.transform.LayerNorm.weight",
            "decoder.cls.predictions.transform.LayerNorm.bias",
            "decoder.cls.predictions.decoder.weight",
            "decoder.cls.predictions.decoder.bias",
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


def convert_bigbird_roberta_ckpt_to_pytorch(ckpt_path: str, save_dir: str):
    tf_weights = get_tf_weights_as_numpy(ckpt_path)
    torch_model = convert_bigbird_roberta(tf_weights)
    torch_model.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--tf_ckpt_path", type=str, help="passed to tf.train.list_variables")
    parser.add_argument("--save_dir", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_bigbird_roberta_ckpt_to_pytorch(args.tf_ckpt_path, args.save_dir)

# python3 src/transformers/models/bigbird_pegasus/convert_bigbird_roberta_tf_to_pytorch.py --tf_ckpt_path src/tf_ckpt/bigbird-roberta-arxiv/model.ckpt-300000 --save_dir src/ckpt/bigbird-roberta-arxiv
