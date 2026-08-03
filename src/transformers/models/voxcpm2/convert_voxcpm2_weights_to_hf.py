# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from pathlib import Path

import torch
from safetensors.torch import load_file


def _convert_weight_norm_key(key: str) -> str:
    if key.endswith("weight_g"):
        return key.removesuffix("weight_g") + "parametrizations.weight.original0"
    if key.endswith("weight_v"):
        return key.removesuffix("weight_v") + "parametrizations.weight.original1"
    return key


def convert_voxcpm2_state_dict(
    model_state_dict: dict[str, torch.Tensor],
    audio_vae_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted_state_dict = {}
    source_state_dicts = (
        ("", model_state_dict),
        ("audio_vae.", audio_vae_state_dict),
    )
    for prefix, state_dict in source_state_dicts:
        for key, value in state_dict.items():
            if prefix and not key.startswith(prefix):
                key = prefix + key
            converted_key = _convert_weight_norm_key(key)
            if converted_key in converted_state_dict:
                raise ValueError(f"Duplicate converted key: {converted_key}")
            converted_state_dict[converted_key] = value
    return converted_state_dict


def _load_checkpoint_file(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if checkpoint_path.suffix == ".safetensors":
        state_dict = load_file(checkpoint_path, device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict) or not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError(f"{checkpoint_path} does not contain a tensor state dictionary")
    return state_dict


def _load_source_state_dicts(input_path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    input_path = Path(input_path)
    model_path = next(
        (
            input_path / filename
            for filename in ("model.safetensors", "pytorch_model.bin")
            if (input_path / filename).is_file()
        ),
        None,
    )
    if model_path is None:
        raise FileNotFoundError(f"No model checkpoint found in {input_path}")

    audio_vae_path = next(
        (
            input_path / filename
            for filename in ("audiovae.safetensors", "audiovae.pth")
            if (input_path / filename).is_file()
        ),
        None,
    )
    if audio_vae_path is None:
        raise FileNotFoundError(f"No AudioVAE checkpoint found in {input_path}")

    return _load_checkpoint_file(model_path), _load_checkpoint_file(audio_vae_path)
