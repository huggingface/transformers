# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
import glob
from pathlib import Path
from typing import Any, Dict, cast

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors import safe_open

from transformers import AutoConfig
from transformers.models.colpali import ColPaliForRetrieval, ColPaliProcessor
from transformers.models.colpali.configuration_colpali import ColPaliConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


ORIGINAL_DTYPE = torch.bfloat16
TOLERANCE = 1e-3

ORIGINAL_CONFIG = AutoConfig.from_pretrained(
    "vidore/colpali-v1.2-merged",
    revision="89fd9736194236a1ecb7a9ec9b04f537f6f896af",
)

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
    "value": torch.tensor(
        [
            [
                [-0.0874, 0.0674, 0.2148],
                [-0.0417, 0.0540, 0.2021],
                [-0.0952, 0.0723, 0.1953],
            ],
            [
                [0.0500, 0.0210, 0.0884],
                [0.0530, 0.0267, 0.1196],
                [-0.0708, 0.1089, 0.1631],
            ],
        ],
        dtype=ORIGINAL_DTYPE,
    ),
}
ORIGINAL_QUERY_OUTPUTS_SLICE = {
    "slice": (slice(None), slice(3), slice(3)),
    "value": torch.tensor(
        [
            [
                [0.1631, -0.0227, 0.0962],
                [-0.1108, -0.1147, 0.0334],
                [-0.0496, -0.1108, -0.0525],
            ],
            [
                [0.1650, -0.0200, 0.0967],
                [-0.0879, -0.1108, 0.0613],
                [-0.1260, -0.0630, 0.1157],
            ],
        ],
        dtype=ORIGINAL_DTYPE,
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


def rename_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("custom_text_proj"):
            new_key = key.replace("custom_text_proj", "embedding_proj_layer")
        if key.startswith("model."):
            new_key = key.replace("model.", "vlm.", 1)
        new_state_dict[new_key] = value
    return new_state_dict


def load_original_state_dict(model_id):
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    # tied wieghts so lm.head is not saved. Let's clone to load state dict
    if "lm_head.weight" not in original_state_dict:
        original_state_dict["vlm.language_model.lm_head.weight"] = original_state_dict[
            "model.language_model.model.embed_tokens.weight"
        ].clone()

    return original_state_dict


@torch.no_grad()
def convert_colpali_weights_to_hf(output_dir: str, push_to_hub: bool):
    # Get the device
    device = get_torch_device("auto")
    print(f"Device: {device}")

    # Load the original model's state_dict
    original_state_dict = load_original_state_dict("vidore/colpali-v1.2-merged")

    # Format the state_dict keys
    original_state_dict = rename_state_dict_keys(original_state_dict)

    # Add the extra attributes for the new model
    new_config = {
        "vlm_config": ORIGINAL_CONFIG.copy(),
        "model_type": "colpali",
        "is_composition": False,
        "embedding_dim": 128,
        "initializer_range": 0.02,  # unused as initialized weights will be replaced
    }

    # Create the new config
    config = cast(ColPaliConfig, ColPaliConfig.from_dict(new_config))

    # Load the untrained model
    model = ColPaliForRetrieval(config=config).to(device).eval()
    print("Created model with new config and randomly initialized weights")

    # NOTE: The model was initialized with float32 weights. We need to convert it to the desired precision.
    # There are two ways to set the model's dtype:
    # - Using `model.from_pretrained(..., torch_dtype=dtype_precision)` doesn't convert the hyperparameters to the desired precision.
    # - Using `model.to(dtype_precision)` converts all values - including the hyperparameters - to the desired precision.
    # The following snippet allows a fine-grained control over the model's dtype, making sure that all
    # the new weights' dtypes match the original model.
    for param in model.parameters():
        param.data = param.data.to(ORIGINAL_DTYPE)
    print(f"Converted the new model weights to `{ORIGINAL_DTYPE}`")

    # Load the original weights
    model.load_state_dict(original_state_dict)
    print("Loaded original model weights")

    # Tie the weights (following ColPali's `__init__`` step)
    if model.vlm.language_model._tied_weights_keys is not None:
        model._tied_weights_keys = [f"vlm.language_model.{k}" for k in model.vlm.language_model._tied_weights_keys]

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
            outputs_images_new[ORIGINAL_IMAGE_OUTPUTS_SLICE["slice"]]
            - ORIGINAL_IMAGE_OUTPUTS_SLICE["value"].to(outputs_images_new.device).to(ORIGINAL_DTYPE)
        )
    )
    mae_queries = torch.mean(
        torch.abs(
            outputs_queries_new[ORIGINAL_QUERY_OUTPUTS_SLICE["slice"]]
            - ORIGINAL_QUERY_OUTPUTS_SLICE["value"].to(outputs_queries_new.device).to(ORIGINAL_DTYPE)
        )
    )

    # Sanity checks
    print(f"Mean Absolute Error (MAE) for images: {mae_images}")
    print(f"Mean Absolute Error (MAE) for queries: {mae_queries}")  # FIXME: MAE ≈ 0.0017
    if mae_images > TOLERANCE or mae_queries > TOLERANCE:
        raise ValueError("Mean Absolute Error (MAE) is greater than the tolerance")

    if not torch.allclose(
        outputs_images_new[ORIGINAL_IMAGE_OUTPUTS_SLICE["slice"]],
        ORIGINAL_IMAGE_OUTPUTS_SLICE["value"].to(outputs_images_new.device).to(ORIGINAL_DTYPE),
        rtol=TOLERANCE,
    ):
        raise ValueError("Outputs for images do not match the original model's outputs")
    if not torch.allclose(
        outputs_queries_new[ORIGINAL_QUERY_OUTPUTS_SLICE["slice"]],
        ORIGINAL_QUERY_OUTPUTS_SLICE["value"].to(outputs_queries_new.device).to(ORIGINAL_DTYPE),
        rtol=TOLERANCE,
    ):
        raise ValueError("Outputs for queries do not match the original model's outputs")

    # Save the model
    if push_to_hub:
        model.push_to_hub(output_dir, private=True)
        print(f"Model pushed to the hub at `{output_dir}`")
    else:
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        print(f"Model saved to `{output_dir}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        This script converts the original ColPali model to the HF model format.
        Example usage: python src/transformers/models/colpali/convert_colpali_weights_to_hf.py --output_dir vidore/colpali-v1.2-hf --push_to_hub".
        """
    )
    parser.add_argument(
        "--output_dir",
        default="vidore/colpali-v1.2-hf",
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
