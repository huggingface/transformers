# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert RoBERTa-PreLayerNorm checkpoint."""


import argparse

import torch

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, RobertaPreLayerNormConfig, RobertaPreLayerNormForMaskedLM
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def convert_roberta_prelayernorm_checkpoint_to_pytorch(checkpoint_repo: str, pytorch_dump_folder_path: str):
    """
    Copy/paste/tweak roberta_prelayernorm's weights to our BERT structure.
    """
    # convert configuration
    config = RobertaPreLayerNormConfig.from_pretrained(
        checkpoint_repo, architectures=["RobertaPreLayerNormForMaskedLM"]
    )

    # convert state_dict
    original_state_dict = torch.load(hf_hub_download(repo_id=checkpoint_repo, filename="pytorch_model.bin"))
    state_dict = {}
    for tensor_key, tensor_value in original_state_dict.items():
        # The transformer implementation gives the model a unique name, rather than overwiriting 'roberta'
        if tensor_key.startswith("roberta."):
            tensor_key = "roberta_prelayernorm." + tensor_key[len("roberta.") :]

        # The original implementation contains weights which are not used, remove them from the state_dict
        if tensor_key.endswith(".self.LayerNorm.weight") or tensor_key.endswith(".self.LayerNorm.bias"):
            continue

        state_dict[tensor_key] = tensor_value

    model = RobertaPreLayerNormForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=None, config=config, state_dict=state_dict
    )
    model.save_pretrained(pytorch_dump_folder_path)

    # convert tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_repo)
    tokenizer.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint-repo",
        default=None,
        type=str,
        required=True,
        help="Path the official PyTorch dump, e.g. 'andreasmadsen/efficient_mlm_m0.40'.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_roberta_prelayernorm_checkpoint_to_pytorch(args.checkpoint_repo, args.pytorch_dump_folder_path)
