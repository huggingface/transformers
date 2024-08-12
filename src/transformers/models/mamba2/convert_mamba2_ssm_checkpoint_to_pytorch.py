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
from os import path
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_model

from transformers import LlamaTokenizerFast, Mamba2Config, Mamba2ForCausalLM, Mamba2TokenizerFast


_MAMBA2_MODELS_DICT = {
    "codestral": {
        "hidden_size": "dim",
        "num_hidden_layers": "n_layers",
        "n_groups": "n_groups",
        "residual_in_fp32": "residual_in_fp32",
        "tie_word_embeddings": "tie_embeddings",
        "norm_before_gate": False,
        "vocab_size": "vocab_size",
        "pad_vocab_size_multiple": "pad_vocab_size_multiple",
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
    },
    "base": {
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layer",
        "n_groups": "ngroups",
        "residual_in_fp32": "residual_in_fp32",
        "tie_word_embeddings": "tie_embeddings",
        "norm_before_gate": False,
        "vocab_size": "vocab_size",
        "pad_vocab_size_multiple": "pad_vocab_size_multiple",
        "bos_token_id": 0,
        "pad_token_id": 0,
        "eos_token_id": 0,
    },
}


def convert_ssm_config_to_hf_config(config_ssm: Dict) -> Tuple[Mamba2Config, bool]:
    """Convert a Mamba2Config from mamba_ssm to a Mamba2Config from here."""
    hf_config = Mamba2Config()

    # Flag for codestral model
    is_not_codestral = "dim" not in config_ssm

    # Switch to a different dict depending on model type
    config_key = "base" if is_not_codestral else "codestral"
    config_dict = _MAMBA2_MODELS_DICT[config_key]

    # Set important values from config and recalculate other resulting entries
    hf_config.hidden_size = config_ssm[config_dict["hidden_size"]]
    hf_config.num_heads = (hf_config.hidden_size * hf_config.expand) // hf_config.head_dim
    hf_config.num_hidden_layers = config_ssm[config_dict["num_hidden_layers"]]
    hf_config.n_groups = config_ssm.get(config_dict["n_groups"], 1)
    hf_config.residual_in_fp32 = config_ssm[config_dict["residual_in_fp32"]]
    hf_config.tie_word_embeddings = config_ssm[config_dict["tie_word_embeddings"]]
    hf_config.norm_before_gate = config_dict["norm_before_gate"]
    hf_config.bos_token_id = config_dict["bos_token_id"]
    hf_config.pad_token_id = config_dict["pad_token_id"]
    hf_config.eos_token_id = config_dict["eos_token_id"]

    # Padded vocab size, mostly of 16 but 32 is also very common in different models
    vocab_size = config_ssm[config_dict["vocab_size"]]
    pad_vocab_size_multiple = config_ssm[config_dict["pad_vocab_size_multiple"]]
    if (vocab_size % pad_vocab_size_multiple) != 0:
        vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
    hf_config.vocab_size = vocab_size

    return hf_config, is_not_codestral


def load_state_dict_from_safetensors(mamba2_checkpoint_path: str) -> Dict[str, torch.Tensor]:
    # Load weights and config from paths
    original_state_dict = {}
    with safe_open(path.join(mamba2_checkpoint_path, "consolidated.safetensors"), framework="pt") as f:
        for k in f.keys():
            newk = k.removeprefix("model.")
            original_state_dict[newk] = f.get_tensor(k).clone()
    return original_state_dict


def load_state_dict_from_torch(mamba2_checkpoint_path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path.join(mamba2_checkpoint_path, "pytorch_model.bin"), map_location="cpu")


def convert_mamba2_checkpoint_file_to_huggingface_model_file(
    mamba2_checkpoint_path: str, precision: str, output_dir: str, tokenizer_model_path: Optional[str] = None
) -> None:
    # Load and save config based on name
    config_path = mamba2_checkpoint_path
    config_path = (
        path.join(config_path, "params.json")
        if path.isfile(path.join(config_path, "params.json"))
        else path.join(config_path, "config.json")
    )
    with open(config_path, "r", encoding="utf-8") as json_file:
        config = json.load(json_file)
    hf_config, is_not_codestral = convert_ssm_config_to_hf_config(config)
    hf_config.save_pretrained(output_dir)

    # Load state dict of the original model
    state_dict_load_function = load_state_dict_from_torch if is_not_codestral else load_state_dict_from_safetensors
    original_state_dict = state_dict_load_function(mamba2_checkpoint_path)

    # Load and transfer to hf model
    hf_model = Mamba2ForCausalLM(hf_config)
    hf_model.load_state_dict(original_state_dict)

    # Save new model to pytorch_dump_path
    dtype = torch.float32 if precision == "fp32" else (torch.bfloat16 if precision == "bf16" else torch.float16)
    save_model(hf_model.to(dtype), path.join(output_dir, "model.safetensors"), metadata={"format": "pt"})

    # Load and save tokenizer
    if tokenizer_model_path is not None and not is_not_codestral:
        tokenizer_class = LlamaTokenizerFast
        tokenizer = tokenizer_class(tokenizer_model_path, legacy=False, from_slow=True)
    else:
        tokenizer = Mamba2TokenizerFast.from_pretrained("state-spaces/mamba-130m-hf")
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--mamba2_checkpoint_directory",
        type=str,
        required=True,
        help="Path to a directory containing the `pytorch_model.bin` or `.safetensors` mamba2_ssm checkpoint file to be converted.",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="bf16",
        required=True,
        choices=["fp32", "fp16", "bf16"],
        help="The precision the model will be saved in. Select from fp32, fp16 or bf16.",
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
