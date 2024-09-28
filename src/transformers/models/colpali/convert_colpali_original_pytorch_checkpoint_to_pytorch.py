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
from PIL import Image

from transformers.models.colpali import ColPaliForRetrieval, ColPaliProcessor
from transformers.models.colpali.configuration_colpali import ColPaliConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


device = get_torch_device("auto")
print(f"Using device: {device}")


def remove_model_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("model."):
            new_key = key[len("model.") :]
        new_state_dict[new_key] = value
    return new_state_dict


def load_original_colpali(device: str = "auto") -> ColPali:
    model = cast(
        ColPali,
        ColPali.from_pretrained(
            "vidore/colpali-v1.2-merged",
            torch_dtype=torch.bfloat16,
            device_map=device,
        ),
    ).eval()
    return model


@torch.no_grad()
def convert_colpali_checkpoint(pytorch_dump_folder_path: str):
    # Load the original model and state_dict
    colpali_original = load_original_colpali(device=device)
    state_dict = colpali_original.state_dict()

    # Format the state_dict keys
    state_dict = remove_model_prefix(state_dict)

    # Load the original config
    original_config = colpali_original.config.to_dict()

    # Add the extra attributes for the new model
    new_config = original_config.copy()
    new_config["model_type"] = "colpali"
    new_config["is_composition"] = False
    new_config["embedding_dim"] = 128

    # Create the new config
    config = cast(ColPaliConfig, ColPaliConfig.from_dict(new_config))

    # Load the untrained model
    model = ColPaliForRetrieval(config=config).to(device).to(torch.bfloat16).eval()
    print("Created model with new config and randomly initialized weights")

    # Load the original weights
    model.load_state_dict(state_dict)
    print("Loaded original model weights")

    # Sanity check: ensure all keys are the same
    state_dict_keys_old = set(state_dict.keys())
    state_dict_keys_new = set(model.state_dict().keys())
    disjoint_keys = state_dict_keys_old.symmetric_difference(state_dict_keys_new)
    if disjoint_keys:
        raise ValueError(f"Incompatible keys: {disjoint_keys}")

    # Sanity checks: forward pass with images and queries
    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (16, 16), color="black"),
    ]
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained("vidore/colpali-v1.2-hf"))

    batch_queries = processor.process_queries(queries).to(device)
    batch_images = processor.process_images(images).to(device)

    with torch.no_grad():
        outputs_images_original = colpali_original(**batch_images)
        breakpoint()
        outputs_images_new = model(**batch_images, return_dict=True).embeddings
        if outputs_images_original.shape != outputs_images_new.shape:
            raise ValueError("Output shapes do not match for images forward pass")
        # FIXME: doesn't match
        if not torch.allclose(outputs_images_original, outputs_images_new, atol=1e-2):
            raise ValueError("Output values do not match for images forward pass")

    with torch.no_grad():
        outputs_queries_original = colpali_original(**batch_queries.copy())
        outputs_queries_new = model(**batch_queries.copy(), return_dict=True).embeddings
        if outputs_queries_original.shape != outputs_queries_new.shape:
            raise ValueError("Output shapes do not match for query forward pass")
        # FIXME: doesn't match
        if not torch.allclose(outputs_queries_original, outputs_queries_new, atol=1e-2):
            raise ValueError("Output values do not match for query forward pass")

    # Save the model
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True, parents=True)
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Model saved to `{pytorch_dump_folder_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pytorch_dump_folder_path", default="checkpoints/colpali", type=str, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()

    convert_colpali_checkpoint(args.pytorch_dump_folder_path)
