# coding=utf-8
# Copyright 2024 state-spaces/mamba2 org and HuggingFace Inc. team.
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
"""This script can be used to convert checkpoints provided in the `mamba2_ssm` library into the format provided in HuggingFace `transformers`. It depends on the `mamba2_ssm` package to be installed."""

import argparse
import json
from os import listdir, path
from typing import Dict, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_model

from transformers import AutoTokenizer, LlamaTokenizerFast, Mamba2Config, Mamba2ForCausalLM


def convert_ssm_config_to_hf_config(config_ssm: Dict) -> Mamba2Config:
    """Convert a Mamba2Config from mamba_ssm to a Mamba2Config from here."""
    hf_config = Mamba2Config()

    # Set important values from config and recalculate other resulting entries
    hf_config.hidden_size = config_ssm["d_model"] if "d_model" in config_ssm else config_ssm["dim"]
    hf_config.num_heads = (hf_config.hidden_size * hf_config.expand) // hf_config.head_dim
    hf_config.num_hidden_layers = config_ssm["n_layer"] if "n_layer" in config_ssm else config_ssm["n_layers"]
    hf_config.n_groups = config_ssm.get("n_groups", 1)
    hf_config.residual_in_fp32 = config_ssm["residual_in_fp32"]
    hf_config.tie_word_embeddings = config_ssm["tie_embeddings"]
    hf_config.pad_token_id = hf_config.bos_token_id = hf_config.eos_token_id = 0
    hf_config.norm_before_gate = False

    # Padded vocab size, mostly of 16 but 32 is also very common in different models
    vocab_size = config_ssm["vocab_size"]
    pad_vocab_size_multiple = config_ssm["pad_vocab_size_multiple"]
    if (vocab_size % pad_vocab_size_multiple) != 0:
        vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
    hf_config.vocab_size = vocab_size

    return hf_config


def load_state_dict_from_safetensors(mamba2_checkpoint_path: str) -> Dict[str, torch.Tensor]:
    # Load weights and config from paths
    original_state_dict = {}
    with safe_open(mamba2_checkpoint_path, framework="pt") as f:
        for k in f.keys():
            newk = k.removeprefix("model.")
            original_state_dict[newk] = f.get_tensor(k).clone()
    return original_state_dict


def load_state_dict_from_torch(mamba2_checkpoint_path: str) -> Dict[str, torch.Tensor]:
    return torch.load(mamba2_checkpoint_path + "/pytorch_model.bin", map_location="cpu")


def convert_mamba2_checkpoint_file_to_huggingface_model_file(
    mamba2_checkpoint_path: str, precision: str, output_dir: str, tokenizer_model_path: Optional[str] = None
) -> None:
    # Load and save config based on name
    config_path = mamba2_checkpoint_path
    config_path = (
        config_path + "/params.json" if path.isfile(config_path + "/params.json") else config_path + "/config.json"
    )
    with open(config_path, "r", encoding="utf-8") as json_file:
        config = json.load(json_file)
    hf_config = convert_ssm_config_to_hf_config(config)
    hf_config.save_pretrained(output_dir)

    # Check the type of the state dict it was saved in
    is_safetensors = False
    for file_name in listdir(mamba2_checkpoint_path):
        if file_name.endswith(".safetensors"):
            is_safetensors = True
            break

    # Load state dict of the original model
    state_dict_load_function = load_state_dict_from_safetensors if is_safetensors else load_state_dict_from_torch
    original_state_dict = state_dict_load_function(mamba2_checkpoint_path)

    # Load and transfer to hf model
    hf_model = Mamba2ForCausalLM(hf_config)
    hf_model.load_state_dict(original_state_dict)

    # Save new model to pytorch_dump_path
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    save_model(hf_model.to(dtype), output_dir + "/model.safetensors", metadata={"format": "pt"})

    # Load and save tokenizer
    if tokenizer_model_path is not None:
        tokenizer_class = LlamaTokenizerFast
        tokenizer = tokenizer_class(tokenizer_model_path, legacy=False, from_slow=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        # TODO: how to correctly overwrite padding side so it is saved as is
        tokenizer.padding_side = "left"
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--mamba2_checkpoint_file",
        type=str,
        required=True,
        help="Path to a `pytorch_model.bin` or `.safetensors` mamba2_ssm checkpoint file to be converted.",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="bf16",
        required=True,
        choices=["fp32", "bf16"],
        help="The precision the model will be saved in. Select from fp32 or bf16.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to directory to save the converted output model to."
    )
    parser.add_argument(
        "-t",
        "--tokenizer_model_path",
        type=str,
        default=None,
        required=False,
        help="Path to a tokenizer file.",
    )
    args = parser.parse_args()

    convert_mamba2_checkpoint_file_to_huggingface_model_file(
        args.mamba2_checkpoint_file, args.precision, args.output_dir, args.tokenizer_model_path
    )
