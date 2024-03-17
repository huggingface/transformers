# coding=utf-8
# Copyright 2024 state-spaces/mamba org and HuggingFace Inc. team.
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
"""This script can be used to convert checkpoints provided in the `mamba_ssm` library into the format provided in HuggingFace `transformers`. It depends on the `mamba_ssm` package to be installed."""

import argparse
import json

import torch

from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM
from transformers.utils import logging
from transformers.utils.import_utils import is_mamba_ssm_available


if is_mamba_ssm_available():
    from mamba_ssm.models.config_mamba import MambaConfig as MambaConfig_ssm

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def convert_ssm_config_to_hf_config(config_ssm: MambaConfig_ssm) -> MambaConfig:
    """Convert a MambaConfig from mamba_ssm to a MambaConfig from transformers."""
    hf_config = MambaConfig()
    # Set config hidden size, num hidden layers, and vocab size directly from the original config
    hf_config.hidden_size = config_ssm.d_model
    hf_config.num_hidden_layers = config_ssm.n_layer
    vocab_size = config_ssm.vocab_size
    pad_vocab_size_multiple = config_ssm.pad_vocab_size_multiple
    if (vocab_size % pad_vocab_size_multiple) != 0:
        vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
    hf_config.vocab_size = vocab_size
    return hf_config


def convert_mamba_ssm_checkpoint_to_pytorch(mamba_checkpoint_path, config_file, output_dir):
    if not is_mamba_ssm_available():
        raise ImportError(
            "Calling convert_mamba_ssm_checkpoint_to_pytorch requires the mamba_ssm library to be installed. Please install it with `pip install mamba_ssm`."
        )
    logger.info(f"Loading model from {mamba_checkpoint_path} based on config from {config_file}")
    # Load config from path
    with open(config_file, "r", encoding="utf-8") as json_file:
        original_ssm_config_json = json.load(json_file)
    original_ssm_config = MambaConfig_ssm(**original_ssm_config_json)

    # Convert mamba_ssm config to huggingface MambaConfig
    hf_config = convert_ssm_config_to_hf_config(original_ssm_config)

    # Load model checkpoint from mamba_checkpoint_path
    original_state_dict = torch.load(mamba_checkpoint_path, map_location="cpu")
    # Rename weights
    original_state_dict["backbone.embeddings.weight"] = original_state_dict["backbone.embedding.weight"]
    original_state_dict.pop("backbone.embedding.weight")

    # Load reshaped state dict into a huggingface model.
    hf_model = MambaForCausalLM(hf_config)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    hf_model.load_state_dict(original_state_dict)

    # Save new model to pytorch_dump_path
    hf_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--mamba_checkpoint_file",
        type=str,
        required=True,
        help="Path to a `pytorch_model.bin` mamba_ssm checkpoint file to be converted.",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help="Path to a `config.json` file corresponding to a MambaConfig of the original mamba_ssm model.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to directory to save the converted output model to."
    )
    args = parser.parse_args()

    convert_mamba_ssm_checkpoint_to_pytorch(args.mamba_checkpoint_file, args.config_file, args.output_dir)
