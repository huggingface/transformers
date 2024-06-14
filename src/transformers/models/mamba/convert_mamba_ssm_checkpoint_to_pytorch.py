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
import math
from typing import Tuple

import torch

from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM
from transformers.utils import logging
from transformers.utils.import_utils import is_mamba_ssm_available


if is_mamba_ssm_available():
    from mamba_ssm.models.config_mamba import MambaConfig as MambaConfigSSM
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    def convert_ssm_config_to_hf_config(config_ssm: MambaConfigSSM) -> MambaConfig:
        """Convert a MambaConfig from mamba_ssm to a MambaConfig from transformers."""
        hf_config = MambaConfig()
        # Set config hidden size, num hidden layers, and vocab size directly from the original config
        hf_config.hidden_size = config_ssm.d_model
        hf_config.intermediate_size = config_ssm.d_model * 2
        hf_config.time_step_rank = math.ceil(config_ssm.d_model / 16)

        hf_config.num_hidden_layers = config_ssm.n_layer
        vocab_size = config_ssm.vocab_size
        pad_vocab_size_multiple = config_ssm.pad_vocab_size_multiple
        if (vocab_size % pad_vocab_size_multiple) != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        hf_config.vocab_size = vocab_size
        return hf_config


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def convert_mamba_ssm_checkpoint_to_huggingface_model(
    original_state_dict: dict, original_ssm_config_dict: dict
) -> Tuple[MambaForCausalLM, AutoTokenizer]:
    if not is_mamba_ssm_available():
        raise ImportError(
            "Calling convert_mamba_ssm_checkpoint_to_huggingface_model requires the mamba_ssm library to be installed. Please install it with `pip install mamba_ssm`."
        )
    original_ssm_config = MambaConfigSSM(**original_ssm_config_dict)

    # Convert mamba_ssm config to huggingface MambaConfig
    hf_config = convert_ssm_config_to_hf_config(original_ssm_config)

    # No weights need to be renamed between the two models.
    converted_state_dict = original_state_dict

    # Load reshaped state dict into a huggingface model.
    hf_model = MambaForCausalLM(hf_config)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    hf_model.load_state_dict(converted_state_dict)
    return (hf_model, tokenizer)


def validate_converted_model(
    original_state_dict: dict, original_ssm_config_dict: dict, hf_model: MambaForCausalLM, tokenizer: AutoTokenizer
) -> None:
    """Validate the converted model returns the same output as the original model."""
    torch_device = "cuda"

    original_config = MambaConfigSSM(**original_ssm_config_dict)
    original_model = MambaLMHeadModel(original_config).to(torch_device)
    original_model.load_state_dict(original_state_dict)

    hf_model = hf_model.to(torch_device)
    input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"].to(torch_device)
    # Assert model logits are close
    with torch.no_grad():
        original_model_logits = original_model(input_ids).logits
        hf_model_logits = hf_model(input_ids).logits
    if not torch.allclose(original_model_logits, hf_model_logits, atol=1e-3):
        raise ValueError("The converted model did not return the same logits as the original model.")

    logger.info("Model conversion validated successfully.")


def convert_mamba_checkpoint_file_to_huggingface_model_file(
    mamba_checkpoint_path: str, config_json_file: str, output_dir: str
) -> None:
    if not is_mamba_ssm_available():
        raise ImportError(
            "Calling convert_mamba_checkpoint_file_to_huggingface_model_file requires the mamba_ssm library to be installed. Please install it with `pip install mamba_ssm`."
        )
    if not torch.cuda.is_available():
        raise ValueError(
            "This script is to be run with a CUDA device, as the original mamba_ssm model does not support cpu."
        )
    logger.info(f"Loading model from {mamba_checkpoint_path} based on config from {config_json_file}")
    # Load weights and config from paths
    original_state_dict = torch.load(mamba_checkpoint_path, map_location="cpu")
    with open(config_json_file, "r", encoding="utf-8") as json_file:
        original_ssm_config_dict = json.load(json_file)

    # Convert the model
    hf_model, tokenizer = convert_mamba_ssm_checkpoint_to_huggingface_model(
        original_state_dict, original_ssm_config_dict
    )

    # Validate the conversion
    validate_converted_model(original_state_dict, original_ssm_config_dict, hf_model, tokenizer)

    logger.info(f"Model converted successfully. Saving model to {output_dir}")

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
        "--config_json_file",
        type=str,
        required=True,
        help="Path to a `config.json` file corresponding to a MambaConfig of the original mamba_ssm model.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to directory to save the converted output model to."
    )
    args = parser.parse_args()

    convert_mamba_checkpoint_file_to_huggingface_model_file(
        args.mamba_checkpoint_file, args.config_json_file, args.output_dir
    )
