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
r"""
Convert RoMa checkpoints to the HuggingFace format.

RoMa, MatchAnything-RoMa and MINIMA-RoMa all share the same RoMa-v1 architecture, so a single config + module tree
loads all of them; only the checkpoint URL (and a couple of config presets) change. The DINOv2 backbone weights are
*not* stored in the RoMa checkpoint (the original code keeps the frozen backbone outside the state dict), so they are
loaded separately from `facebook/dinov2-large`.

Example:

```bash
python src/transformers/models/roma/convert_roma_to_hf.py \
    --checkpoint roma_outdoor \
    --pytorch_dump_folder_path /tmp/roma_outdoor --save_model
```
"""

import argparse
import re

import torch
from huggingface_hub import hf_hub_download

from transformers import Dinov2Backbone, RomaConfig, RomaForKeypointMatching, RomaImageProcessor


# Each checkpoint shares the RoMa-v1 architecture; only the weights (and a couple of config presets) change.
CHECKPOINTS = {
    "roma_outdoor": {
        "url": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "config": {},
    },
    "roma_indoor": {
        "url": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
        "config": {},
    },
    # MatchAnything and MINIMA publish RoMa-architecture weights trained on more diverse / multimodal data.
    "matchanything_roma": {
        "repo_id": "vismatch/matchanything-roma",
        "filename": "model.safetensors",
        "config": {},
    },
    "minima_roma": {
        "url": "https://github.com/LSXI7/storage/releases/download/MINIMA/minima_roma.pth",
        "config": {},
    },
}

# Regex key remapping from the original `RegressionMatcher` state dict to `RomaForKeypointMatching`. The decoder
# submodule names were chosen to match the original, so only the encoder CNN prefix needs rewriting. The leading
# `matcher.model.` strip handles the MatchAnything checkpoints, which nest the RoMa weights.
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^matcher\.model\.": "",
    r"^encoder\.cnn\.": "roma.cnn.",
}


def convert_old_keys_to_new_keys(state_dict: dict) -> dict:
    """Apply the regex remapping and drop the (absent-from-HF) frozen DINOv2 backbone keys, if any."""
    new_state_dict = {}
    for key, value in state_dict.items():
        # The frozen DINOv2 backbone is loaded separately; ignore any backbone keys that may be present.
        if key.startswith("encoder.dinov2") or "dinov2_vitl14" in key:
            continue
        new_key = key
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            new_key = re.sub(pattern, replacement, new_key)
        new_state_dict[new_key] = value
    return new_state_dict


def pad_conv_refiner_weights(state_dict: dict, model_state_dict: dict) -> dict:
    """
    The HF model uses RoMa's padded `ConvRefiner` channels so that every checkpoint loads into one module tree.
    Non-padded checkpoints have smaller `conv_refiner` tensors; zero-pad them to the model's expected shapes (this
    mirrors `roma_models.pad_refiner_state_dict`).
    """
    for key, target in model_state_dict.items():
        if key.startswith("decoder.conv_refiner") and key in state_dict:
            source = state_dict[key]
            if source.shape != target.shape:
                padded = torch.zeros(target.shape, dtype=source.dtype)
                padded[tuple(slice(0, s) for s in source.shape)] = source
                state_dict[key] = padded
    return state_dict


def load_original_state_dict(checkpoint: str) -> dict:
    spec = CHECKPOINTS[checkpoint]
    if "url" in spec:
        state_dict = torch.hub.load_state_dict_from_url(spec["url"], map_location="cpu")
    else:
        path = hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"])
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
    # Checkpoints may wrap the weights under "state_dict"/"model".
    for wrapper_key in ("state_dict", "model", "weights"):
        if isinstance(state_dict, dict) and wrapper_key in state_dict and isinstance(state_dict[wrapper_key], dict):
            state_dict = state_dict[wrapper_key]
            break
    return state_dict


@torch.no_grad()
def write_model(checkpoint: str, output_path: str, push_to_hub: bool = False, organization: str | None = None) -> None:
    # The published RoMa models run the high-resolution refinement pass, so enable it on the converted checkpoints.
    # `model(**processor(images))` then reproduces the reference behaviour (the processor also returns the high-res
    # `pixel_values_upsampled`); the pass degrades gracefully to coarse-only if that input is omitted.
    config = RomaConfig(upsample_predictions=True, **CHECKPOINTS[checkpoint]["config"])
    config.architectures = ["RomaForKeypointMatching"]
    model = RomaForKeypointMatching(config).eval()

    # 1) Load everything except the frozen DINOv2 backbone from the RoMa checkpoint.
    original_state_dict = load_original_state_dict(checkpoint)
    converted_state_dict = convert_old_keys_to_new_keys(original_state_dict)
    converted_state_dict = pad_conv_refiner_weights(converted_state_dict, model.state_dict())

    # 2) Load the frozen DINOv2 backbone from the canonical HF checkpoint.
    backbone = Dinov2Backbone.from_pretrained("facebook/dinov2-large", out_indices=[-1])
    for name, value in backbone.state_dict().items():
        converted_state_dict[f"roma.backbone.{name}"] = value

    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    # Buffers (e.g. BatchNorm `num_batches_tracked`) and the parameterless cosine kernel may show up as missing.
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing:
        print(f"[warn] missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    image_processor = RomaImageProcessor(do_upsample=True)

    if output_path is not None:
        model.save_pretrained(output_path)
        image_processor.save_pretrained(output_path)
    if push_to_hub:
        repo_id = f"{organization}/{checkpoint}" if organization else checkpoint
        model.push_to_hub(repo_id)
        image_processor.push_to_hub(repo_id)
    print(f"Converted '{checkpoint}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=list(CHECKPOINTS), required=True)
    parser.add_argument("--pytorch_dump_folder_path", dest="output_path", default=None)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--organization", default=None)
    args = parser.parse_args()
    write_model(
        checkpoint=args.checkpoint,
        output_path=args.output_path if args.save_model else None,
        push_to_hub=args.push_to_hub,
        organization=args.organization,
    )


if __name__ == "__main__":
    main()
