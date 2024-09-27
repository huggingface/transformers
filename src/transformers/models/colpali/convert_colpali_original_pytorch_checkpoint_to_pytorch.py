# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert ColPali checkpoint."""

import argparse
from pathlib import Path
from typing import Any, Dict, cast

import torch
from colpali_engine.models import ColPali
from colpali_engine.utils.torch_utils import get_torch_device

from transformers.models.colpali.configuration_colpali import ColPaliConfig
from transformers.models.colpali.modeling_colpali import ColPaliForRetrieval
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def remove_model_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("model."):
            new_key = key[len("model.") :]
        new_state_dict[new_key] = value
    return new_state_dict


def load_original_colpali() -> ColPali:
    model = cast(
        ColPali,
        ColPali.from_pretrained(
            "vidore/colpali-v1.2-merged",
            torch_dtype=torch.bfloat16,
            device_map=get_torch_device("auto"),
        ),
    )
    return model


@torch.no_grad()
def convert_colpali_checkpoint(pytorch_dump_folder_path: str):
    # Load the original model and state_dict
    colpali_original = load_original_colpali()
    state_dict = colpali_original.state_dict()

    # Format the state_dict keys
    state_dict = remove_model_prefix(state_dict)

    # Load the original config
    original_config = colpali_original.config.to_dict()

    # Add the extra attributes for the new model
    new_config = original_config.copy()
    new_config["embedding_dim"] = 128

    # Create the new config
    config = cast(ColPaliConfig, ColPaliConfig.from_dict(new_config))

    # Load the untrained model
    model = ColPaliForRetrieval(config=config).to(torch.bfloat16).eval()
    print("Created model with new config and randomly initialized weights")

    # Load the original weights
    model.load_state_dict(state_dict)
    print("Loaded original model weights")

    # Save the model
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True, parents=True)
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Model saved to `{pytorch_dump_folder_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()

    convert_colpali_checkpoint(args.pytorch_dump_folder_path)
