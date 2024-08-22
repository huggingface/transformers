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

import torch
from safetensors import safe_open

from transformers import LlamaTokenizerFast, Mamba2Config, Mamba2ForCausalLM


def convert_mamba2_checkpoint_file_to_huggingface_model_file(
    mamba2_checkpoint_path: str, tokenizer_model_path: str, output_dir: str
) -> None:
    hf_config = Mamba2Config()
    hf_model = Mamba2ForCausalLM(hf_config)
    # Load weights and config from paths
    original_state_dict = {}
    with safe_open(mamba2_checkpoint_path, framework="pt") as f:
        for k in f.keys():
            newk = k.removeprefix("model.")
            original_state_dict[newk] = f.get_tensor(k).clone()

    hf_model.load_state_dict(original_state_dict)

    # Save new model to pytorch_dump_path
    hf_model.to(torch.bfloat16).save_pretrained(output_dir)
    tokenizer_class = LlamaTokenizerFast
    tokenizer = tokenizer_class(tokenizer_model_path, legacy=False, from_slow=True)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--mamba2_checkpoint_file",
        type=str,
        required=True,
        help="Path to a `pytorch_model.bin` mamba2_ssm checkpoint file to be converted.",
    )
    parser.add_argument(
        "-c",
        "--tokenizer_model_path",
        type=str,
        required=True,
        help="Path to a `config.json` file corresponding to a Mamba2Config of the original mamba2_ssm model.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to directory to save the converted output model to."
    )
    args = parser.parse_args()

    convert_mamba2_checkpoint_file_to_huggingface_model_file(
        args.mamba2_checkpoint_file, args.tokenizer_model_path, args.output_dir
    )
