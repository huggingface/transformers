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
"""Convert ColPali weights."""

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


ORIGINAL_DTYPE = torch.bfloat16
TOLERANCE = 1e-2


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

TEST_IMAGES = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]
TEST_QUERIES = [
    "What is the organizational structure for our R&D department?",
    "Can you provide a breakdown of last year’s financial performance?",
]

ORIGINAL_IMAGE_OUTPUTS_SLICE = {
    "slice": (slice(None), slice(3), slice(3)),
    "value": torch.FloatTensor(
        [
            [
                [-0.06103515625, 0.0849609375, 0.1943359375],
                [-0.052001953125, 0.0859375, 0.125],
                [-0.08740234375, 0.0703125, 0.189453125],
            ],
            [
                [0.043212890625, 0.0211181640625, 0.06689453125],
                [0.046142578125, 0.01422119140625, 0.1416015625],
                [-0.07421875, 0.103515625, 0.1669921875],
            ],
        ]
    ),
}
ORIGINAL_QUERY_OUTPUTS_SLICE = {
    "slice": (slice(None), slice(3), slice(3)),
    "value": torch.FloatTensor(
        [
            [
                [0.162109375, -0.0206298828125, 0.09716796875],
                [-0.107421875, -0.1162109375, 0.028076171875],
                [-0.0458984375, -0.1123046875, -0.055908203125],
            ],
            [
                [0.1650390625, -0.019775390625, 0.0966796875],
                [-0.09228515625, -0.11181640625, 0.06396484375],
                [-0.1298828125, -0.06396484375, 0.1171875],
            ],
        ]
    ),
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
    original_state_dict: Dict[str, torch.Tensor] = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/vidore/colpali-v1.2-merged-state_dict/resolve/main/colpali_v1_2_merged_state_dict.pth",
        map_location="cpu",
    )

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
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained("vidore/colpali-v1.2-merged"))

    batch_images = processor.process_images(images=TEST_IMAGES).to(device)
    batch_queries = processor.process_queries(text=TEST_QUERIES).to(device)

    # Predict with the new model
    with torch.no_grad():
        outputs_images_new = model(**batch_images, return_dict=True).embeddings
        outputs_queries_new = model(**batch_queries, return_dict=True).embeddings

    # Compare the outputs with the original model
    mae_images = torch.mean(
        torch.abs(
            outputs_images_new[ORIGINAL_IMAGE_OUTPUTS_SLICE["slice"]].to(ORIGINAL_DTYPE)
            - ORIGINAL_IMAGE_OUTPUTS_SLICE["value"].to(outputs_images_new.device).to(ORIGINAL_DTYPE)
        )
    )
    mae_queries = torch.mean(
        torch.abs(
            outputs_queries_new[ORIGINAL_QUERY_OUTPUTS_SLICE["slice"]].to(ORIGINAL_DTYPE)
            - ORIGINAL_QUERY_OUTPUTS_SLICE["value"].to(outputs_queries_new.device).to(ORIGINAL_DTYPE)
        )
    )

    print(f"Mean Absolute Error (MAE) for images: {mae_images}")
    print(f"Mean Absolute Error (MAE) for queries: {mae_queries}")

    if mae_images > TOLERANCE or mae_queries > TOLERANCE:
        raise ValueError("Mean Absolute Error (MAE) is greater than the tolerance")

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
