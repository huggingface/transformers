# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert MRA checkpoints from the original repository. URL: https://github.com/mlpen/mra-attention"""

import argparse

import torch

from transformers import MraConfig, MraForMaskedLM


def rename_key(orig_key):
    if "model" in orig_key:
        orig_key = orig_key.replace("model.", "")
    if "norm1" in orig_key:
        orig_key = orig_key.replace("norm1", "attention.output.LayerNorm")
    if "norm2" in orig_key:
        orig_key = orig_key.replace("norm2", "output.LayerNorm")
    if "norm" in orig_key:
        orig_key = orig_key.replace("norm", "LayerNorm")
    if "transformer" in orig_key:
        layer_num = orig_key.split(".")[0].split("_")[-1]
        orig_key = orig_key.replace(f"transformer_{layer_num}", f"encoder.layer.{layer_num}")
    if "mha.attn" in orig_key:
        orig_key = orig_key.replace("mha.attn", "attention.self")
    if "mha" in orig_key:
        orig_key = orig_key.replace("mha", "attention")
    if "W_q" in orig_key:
        orig_key = orig_key.replace("W_q", "self.query")
    if "W_k" in orig_key:
        orig_key = orig_key.replace("W_k", "self.key")
    if "W_v" in orig_key:
        orig_key = orig_key.replace("W_v", "self.value")
    if "ff.0" in orig_key:
        orig_key = orig_key.replace("ff.0", "intermediate.dense")
    if "ff.2" in orig_key:
        orig_key = orig_key.replace("ff.2", "output.dense")
    if "ff" in orig_key:
        orig_key = orig_key.replace("ff", "output.dense")
    if "mlm_class" in orig_key:
        orig_key = orig_key.replace("mlm.mlm_class", "cls.predictions.decoder")
    if "mlm" in orig_key:
        orig_key = orig_key.replace("mlm", "cls.predictions.transform")
    if "backbone.backbone.encoders" in orig_key:
        orig_key = orig_key.replace("backbone.backbone.encoders", "encoder.layer")
    if "cls" not in orig_key:
        orig_key = "mra." + orig_key

    return orig_key


def convert_checkpoint_helper(max_position_embeddings, orig_state_dict):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if ("pooler" in key) or ("sen_class" in key):
            continue
        else:
            orig_state_dict[rename_key(key)] = val

    orig_state_dict["cls.predictions.bias"] = orig_state_dict["cls.predictions.decoder.bias"]
    orig_state_dict["mra.embeddings.position_ids"] = torch.arange(max_position_embeddings).expand((1, -1)) + 2

    return orig_state_dict


def convert_mra_checkpoint(checkpoint_path, mra_config_file, pytorch_dump_path):
    orig_state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
    config = MraConfig.from_json_file(mra_config_file)
    model = MraForMaskedLM(config)

    new_state_dict = convert_checkpoint_helper(config.max_position_embeddings, orig_state_dict)

    print(model.load_state_dict(new_state_dict))
    model.eval()
    model.save_pretrained(pytorch_dump_path)

    print(f"Checkpoint successfuly converted. Model saved at {pytorch_dump_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pytorch_model_path", default=None, type=str, required=True, help="Path to Mra pytorch checkpoint."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The json file for Mra model config.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_mra_checkpoint(args.pytorch_model_path, args.config_file, args.pytorch_dump_path)
