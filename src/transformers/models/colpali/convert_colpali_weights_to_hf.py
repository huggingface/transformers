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
from PIL import Image

from transformers.models.colpali import ColPaliForRetrieval, ColPaliProcessor
from transformers.models.colpali.configuration_colpali import ColPaliConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


ORIGINAL_DTYPE = torch.float16
TOLERANCE = 2e-3


# Copied from https://huggingface.co/vidore/colpali-v1.2-merged/blob/main/config.json
ORIGINAL_CONFIG = {
    "_name_or_path": "vidore/colpaligemma-3b-pt-448-base",
    "architectures": ["ColPali"],
    "bos_token_id": 2,
    "eos_token_id": 1,
    "hidden_size": 2048,
    "ignore_index": -100,
    "image_token_index": 257152,
    "model_type": "paligemma",
    "pad_token_id": 0,
    "projection_dim": 2048,
    "text_config": {
        "hidden_size": 2048,
        "intermediate_size": 16384,
        "model_type": "gemma",
        "num_attention_heads": 8,
        "num_hidden_layers": 18,
        "num_image_tokens": 1024,
        "num_key_value_heads": 1,
        "torch_dtype": "float32",
        "vocab_size": 257216,
    },
    "torch_dtype": "bfloat16",
    "transformers_version": "4.44.0",
    "vision_config": {
        "hidden_size": 1152,
        "image_size": 448,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "num_image_tokens": 1024,
        "patch_size": 14,
        "projection_dim": 2048,
        "projector_hidden_act": "gelu_fast",
        "vision_use_head": False,
    },
}


def get_torch_device(device: str = "auto") -> str:
    """
    Returns the device (string) to be used by PyTorch.

    `device` arg defaults to "auto" which will use:
    - "cuda:0" if available
    - else "mps" if available
    - else "cpu".
    """

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():  # for Apple Silicon
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")

    return device


def remove_model_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("model."):
            new_key = key[len("model.") :]
        new_state_dict[new_key] = value
    return new_state_dict


@torch.no_grad()
def convert_colpali_weights_to_hf(output_dir: str, push_to_hub: bool):
    # Get the device
    device = get_torch_device("auto")
    print(f"Device: {device}")

    # Load the original model's state_dict
    # TODO: replace with new state_dict URL (.pth file)
    original_state_dict: Dict[str, torch.Tensor] = torch.hub.load_state_dict_from_url(
        "vidore/colpali-v1.2-merged",
        map_location="cpu",
    )["model"]

    # Format the state_dict keys
    original_state_dict = remove_model_prefix(original_state_dict)

    # Add the extra attributes for the new model
    new_config = ORIGINAL_CONFIG.copy()
    new_config["model_type"] = "colpali"
    new_config["is_composition"] = False
    new_config["embedding_dim"] = 128

    # Create the new config
    config = cast(ColPaliConfig, ColPaliConfig.from_dict(new_config))

    # Load the untrained model
    model = ColPaliForRetrieval(config=config).to(device).eval()
    print("Created model with new config and randomly initialized weights")

    # NOTE: The model was initialized with float32 weights. We need to convert it to the desired precision.
    # Using `model.to(ORIGINAL_DTYPE)` also converts the hyperparameters to the desired precision, which is not desired.
    # Hence, we need to manually convert the weights to the desired precision.
    for param in model.parameters():
        param.data = param.data.to(ORIGINAL_DTYPE)
    print(f"Converted the new model weights to `{ORIGINAL_DTYPE}`")

    # Load the original weights
    model.load_state_dict(original_state_dict)
    print("Loaded original model weights")

    # Tie the weights (following ColPali's `__init__`` step)
    if model.language_model._tied_weights_keys is not None:
        model._tied_weights_keys = [f"language_model.{k}" for k in model.language_model._tied_weights_keys]

    # Sanity check: ensure all keys are the same
    state_dict_keys_old = set(original_state_dict.keys())
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

    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained("vidore/colpali-v1.2-merged"))

    batch_queries = processor(text=queries).to(device)
    batch_images = processor(images=images).to(device)

    with torch.no_grad():
        outputs_images_new = model(**batch_images, return_dict=True).embeddings
        outputs_queries_new = model(**batch_queries.copy(), return_dict=True).embeddings

    if outputs_images_original.shape != outputs_images_new.shape:
        raise ValueError("Output shapes do not match for images forward pass")

    if outputs_queries_original.shape != outputs_queries_new.shape:
        raise ValueError("Output shapes do not match for query forward pass")

    # Save the model
    if push_to_hub:
        model.push_to_hub(output_dir, private=True)
        print(f"Model pushed to the hub at `{output_dir}`")
    else:
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        model.save_pretrained(output_dir)
        print(f"Model saved to `{output_dir}`")


CLI_HELP = """
This script converts the original ColPali model to the HF model format.\n

Example usage: "python src/transformers/models/colpali/convert_colpali_weights_to_hf.py --output_dir vidore/colpali-v1.2-hf --push_to_hub".
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=CLI_HELP)
    parser.add_argument(
        "--output_dir",
        default="google/gemma-7b",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--push_to_hub",
        help="Whether or not to push the model to the hub at `output_dir` instead of saving it locally.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    convert_colpali_weights_to_hf(output_dir=args.output_dir, push_to_hub=args.push_to_hub)
