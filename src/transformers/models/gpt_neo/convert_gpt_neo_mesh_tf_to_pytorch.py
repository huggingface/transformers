# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team.
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
"""Convert GPT Neo checkpoint."""


import argparse
import json

from transformers import GPTNeoConfig, GPTNeoForCausalLM, load_tf_weights_in_gpt_neo
from transformers.utils import logging


logging.set_verbosity_info()


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config_json = json.load(open(config_file, "r"))
    config = GPTNeoConfig(
        hidden_size=config_json["n_embd"],
        num_layers=config_json["n_layer"],
        num_heads=config_json["n_head"],
        attention_types=config_json["attention_types"],
        max_position_embeddings=config_json["n_positions"],
        resid_dropout=config_json["res_dropout"],
        embed_dropout=config_json["embed_dropout"],
        attention_dropout=config_json["attn_dropout"],
    )
    print(f"Building PyTorch model from configuration: {config}")
    model = GPTNeoForCausalLM(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_gpt_neo(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained mesh-tf model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path)
