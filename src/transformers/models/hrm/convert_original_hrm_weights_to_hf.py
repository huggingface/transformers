# Copyright 2025 Sapient Inc. All rights reserved.
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
"""
Convert HRM checkpoints from the original implementation to HuggingFace format.

This script converts weights from the original HRM implementation to the HuggingFace
transformers format. The HRM architecture was introduced in:
"Hierarchical Reasoning Model" by Guan Wang, Jin Li, Yuhao Sun, Xing Chen,
Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori.
Paper: https://arxiv.org/abs/2506.21734

Sample usage:

```
python src/transformers/models/hrm/convert_original_hrm_weights_to_hf.py \
    --input_checkpoint /path/to/original/checkpoint.pt \
    --output_dir /output/path \
    --config_file /path/to/config.json
```

Thereafter, models can be loaded via:

```python
from transformers import HrmForCausalLM

model = HrmForCausalLM.from_pretrained("/output/path")
```
"""

import argparse
import json
import os

import torch

from transformers import HrmConfig, HrmForCausalLM


def convert_hrm_checkpoint(
    input_checkpoint: str,
    output_dir: str,
    config_file: str | None = None,
    safe_serialization: bool = True,
    push_to_hub: bool = False,
    repo_id: str | None = None,
):
    """
    Convert HRM checkpoint from original format to HuggingFace format.

    Args:
        input_checkpoint (str): Path to the original HRM checkpoint (.pt file).
        output_dir (str): Directory where the converted model will be saved.
        config_file (str | None): Optional path to a JSON config file. If not provided,
            the config will be inferred from the checkpoint.
        safe_serialization (bool): Whether to save using safetensors format (recommended).
        push_to_hub (bool): Whether to push the converted model to HuggingFace Hub.
        repo_id (str | None): Repository ID for pushing to Hub (required if push_to_hub=True).

    Returns:
        None
    """
    print(f"Loading checkpoint from {input_checkpoint}")
    checkpoint = torch.load(input_checkpoint, map_location="cpu")

    # Load or create config
    if config_file is not None:
        print(f"Loading config from {config_file}")
        with open(config_file) as f:
            config_dict = json.load(f)
        config = HrmConfig(**config_dict)
    else:
        # Try to extract config from checkpoint metadata
        if "config" in checkpoint:
            print("Using config from checkpoint")
            config = HrmConfig(**checkpoint["config"])
        else:
            raise ValueError(
                "No config found in checkpoint and no config_file provided. "
                "Please provide a config file with --config_file"
            )

    # Extract state dict from checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Assume the checkpoint is already a state dict
        state_dict = checkpoint

    print(f"Creating HRM model with config: {config}")
    model = HrmForCausalLM(config)

    # Load state dict into model
    # Note: The original implementation may have different naming conventions
    # This conversion assumes the state dict keys match the HF implementation
    # If there are naming differences, add key mapping here
    print("Loading state dict into model")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

    # Save the model
    print(f"Saving converted model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)

    # Save config separately for reference
    config.save_pretrained(output_dir)

    print("Conversion complete!")

    # Push to Hub if requested
    if push_to_hub:
        if repo_id is None:
            raise ValueError("repo_id must be provided when push_to_hub=True")
        print(f"Pushing model to HuggingFace Hub: {repo_id}")
        model.push_to_hub(repo_id)
        print("Model successfully pushed to Hub!")


def main():
    parser = argparse.ArgumentParser(description="Convert HRM checkpoints to HuggingFace format")
    parser.add_argument(
        "--input_checkpoint",
        type=str,
        required=True,
        help="Path to the original HRM checkpoint file (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the converted model will be saved",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Optional path to a JSON config file",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="Whether to save using safetensors format (recommended)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Whether to push the converted model to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Repository ID for pushing to Hub (e.g., 'username/model-name')",
    )

    args = parser.parse_args()

    convert_hrm_checkpoint(
        input_checkpoint=args.input_checkpoint,
        output_dir=args.output_dir,
        config_file=args.config_file,
        safe_serialization=args.safe_serialization,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
