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
from functools import partial
from os import path
from typing import Dict, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_model

from transformers import LlamaTokenizerFast, Mamba2Config, Mamba2ForCausalLM, Mamba2TokenizerFast


def load_state_dict_from_safetensors(mamba2_checkpoint_path: str, ckpt_name: str) -> Dict[str, torch.Tensor]:
    # Load weights and config from paths
    original_state_dict = {}
    with safe_open(path.join(mamba2_checkpoint_path, ckpt_name), framework="pt") as f:
        for k in f.keys():
            newk = k.removeprefix("model.")
            original_state_dict[newk] = f.get_tensor(k).clone()
    return original_state_dict


def load_state_dict_from_torch(mamba2_checkpoint_path: str, ckpt_name: str) -> Dict[str, torch.Tensor]:
    return torch.load(path.join(mamba2_checkpoint_path, ckpt_name), map_location="cpu")


def convert_ssm_config_to_hf_config(config_ssm: Dict, mamba2_model_dict: Dict) -> Mamba2Config:
    """Convert a Mamba2Config from mamba_ssm to a Mamba2Config from here."""
    hf_config = Mamba2Config()

    # Switch to a different dict depending on model type
    config_dict = mamba2_model_dict

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

    return hf_config


def load_and_save_tokenizer(
    mamba2_model_type: str,
    output_dir: str,
    tokenizer_model_path: Optional[str] = None,
) -> None:
    tokenizer = None

    # Load tokenizer
    if tokenizer_model_path is not None and mamba2_model_type == "codestral":
        tokenizer_class = LlamaTokenizerFast
        tokenizer = tokenizer_class(tokenizer_model_path, legacy=False, from_slow=True)
    elif mamba2_model_type == "mamba_ssm":
        tokenizer = Mamba2TokenizerFast.from_pretrained("state-spaces/mamba-130m-hf")

    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)


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
        "config_name": "params.json",
        "load_state_dict": partial(load_state_dict_from_safetensors, ckpt_name="consolidated.safetensors"),
        "load_and_save_tokenizer": partial(load_and_save_tokenizer, "codestral"),
    },
    "mamba_ssm": {
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
        "config_name": "config.json",
        "load_state_dict": partial(load_state_dict_from_torch, ckpt_name="pytorch_model.bin"),
        "load_and_save_tokenizer": partial(load_and_save_tokenizer, "mamba_ssm"),
    },
}


def convert_mamba2_checkpoint_file_to_huggingface_model_file(
    mamba2_checkpoint_path: str,
    mamba2_model_type: str,
    precision: str,
    output_dir: str,
    tokenizer_model_path: Optional[str] = None,
) -> None:
    mamba2_model_dict = _MAMBA2_MODELS_DICT[mamba2_model_type]

    # Load and save config based on name
    config_path = path.join(mamba2_checkpoint_path, mamba2_model_dict["config_name"])
    with open(config_path, "r", encoding="utf-8") as json_file:
        config = json.load(json_file)
    hf_config = convert_ssm_config_to_hf_config(config_ssm=config, mamba2_model_dict=mamba2_model_dict)
    hf_config.save_pretrained(output_dir)

    # Load state dict of the original model and transfer to hf model
    original_state_dict = mamba2_model_dict["load_state_dict"](mamba2_checkpoint_path=mamba2_checkpoint_path)
    hf_model = Mamba2ForCausalLM(hf_config)
    hf_model.load_state_dict(original_state_dict)

    # Save new model to pytorch_dump_path
    dtype = torch.float32 if precision == "fp32" else (torch.bfloat16 if precision == "bf16" else torch.float16)
    save_model(hf_model.to(dtype), path.join(output_dir, "model.safetensors"), metadata={"format": "pt"})

    # Load and save tokenizer
    mamba2_model_dict["load_and_save_tokenizer"](output_dir=output_dir, tokenizer_model_path=tokenizer_model_path)


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
        "-m",
        "--mamba2_model_type",
        type=str,
        default="mamba_ssm",
        const="mamba_ssm",
        required=True,
        choices=("codestral", "mamba_ssm"),
        help="The model type the conversion will be performed on. Can choose from either `codestral` or `mamba_ssm`.",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="fp16",
        const="fp16",
        required=True,
        choices=("fp32", "fp16", "bf16"),
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
        help="Path to a `codestral` tokenizer file.",
    )
    args = parser.parse_args()

    convert_mamba2_checkpoint_file_to_huggingface_model_file(
        args.mamba2_checkpoint_directory,
        args.mamba2_model_type,
        args.precision,
        args.output_dir,
        args.tokenizer_model_path,
    )
