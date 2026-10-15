# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Convert the original Cellpose checkpoints (https://huggingface.co/mouseland/cellpose-sam) to the HF format.

The original state dict keys are loaded as-is: renaming, qkv splitting and relative position resizing are done by the
`cellpose` / `cellpose_dino` conversion mappings in `conversion_mapping.py`.
"""

import argparse
import os
import tempfile

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from transformers import (
    CellposeConfig,
    CellposeDinoConfig,
    CellposeDinoForInstanceSegmentation,
    CellposeForInstanceSegmentation,
    CellposeImageProcessor,
)


CONFIGS = {
    "cpsam": CellposeConfig(),
    "cpsam_v2": CellposeConfig(),
    "cpdino": CellposeDinoConfig(),
    "cpdino-vitb": CellposeDinoConfig(
        hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12
    ),
}


def convert(model_name: str, output_dir: str, push_to_hub_repo: str | None = None):
    config = CONFIGS[model_name]
    model_class = (
        CellposeForInstanceSegmentation if config.model_type == "cellpose" else CellposeDinoForInstanceSegmentation
    )
    state_dict = torch.load(
        hf_hub_download("mouseland/cellpose-sam", model_name), map_location="cpu", weights_only=True
    )
    state_dict = {key.removeprefix("module."): value.contiguous() for key, value in state_dict.items()}

    with tempfile.TemporaryDirectory() as tmp_dir:
        config.save_pretrained(tmp_dir)
        save_file(state_dict, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})
        model, loading_info = model_class.from_pretrained(tmp_dir, output_loading_info=True)
    if loading_info["missing_keys"] or loading_info["unexpected_keys"] or loading_info["mismatched_keys"]:
        raise ValueError(f"Conversion failed: {loading_info}")

    image_processor = CellposeImageProcessor(tile_size=config.image_size)
    model.save_pretrained(output_dir)
    image_processor.save_pretrained(output_dir)
    if push_to_hub_repo is not None:
        model.push_to_hub(push_to_hub_repo)
        image_processor.push_to_hub(push_to_hub_repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=list(CONFIGS), default="cpsam")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--push_to_hub_repo", default=None)
    args = parser.parse_args()
    convert(args.model_name, args.output_dir, args.push_to_hub_repo)
