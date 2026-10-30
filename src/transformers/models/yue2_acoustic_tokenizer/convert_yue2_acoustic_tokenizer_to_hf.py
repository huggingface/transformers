# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
Converts the YuE2 VAE (https://huggingface.co/m-a-p/YuE2-Vae) to a `YuE2AcousticTokenizerModel` checkpoint.

```bash
python src/transformers/models/yue2_acoustic_tokenizer/convert_yue2_acoustic_tokenizer_to_hf.py \
    --checkpoint m-a-p/YuE2-Vae \
    --output_dir yue2-acoustic-tokenizer \
    --push_to_hub bezzam/yue2-acoustic-tokenizer
```
"""

import argparse
import json
import re
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from transformers import EncodecFeatureExtractor, YuE2AcousticTokenizerConfig, YuE2AcousticTokenizerModel


# The original nests `nn.Sequential`s: a residual unit is (snake, conv, snake, conv), an encoder block is three residual
# units then (snake, strided conv), a decoder block is (snake, transposed conv) then three residual units, and the
# encoder and decoder are (conv, six blocks, snake, conv).
# fmt: off
RESIDUAL_UNIT_MAPPING = {"0": "snake1", "1": "conv1", "2": "snake2", "3": "conv2"}
ENCODER_BLOCK_MAPPING = {"0": "res_unit1", "1": "res_unit2", "2": "res_unit3", "3": "snake1", "4": "conv1"}
DECODER_BLOCK_MAPPING = {"0": "snake1", "1": "conv_t1", "2": "res_unit1", "3": "res_unit2", "4": "res_unit3"}
# fmt: on


def convert_key(key: str, num_blocks: int) -> str:
    """Maps a key of the original's nested `nn.Sequential`s to the named modules of the converted model."""
    match = re.fullmatch(r"(encoder|decoder)\.layers\.(\d+)\.(.+)", key)
    if match is None:
        raise ValueError(f"Unexpected key: {key}")
    part, index, rest = match.group(1), int(match.group(2)), match.group(3)
    if index == 0:
        return f"{part}.conv1.{rest}"
    if index == num_blocks + 1:
        return f"{part}.snake1.{rest}"
    if index == num_blocks + 2:
        return f"{part}.conv2.{rest}"

    block_mapping = ENCODER_BLOCK_MAPPING if part == "encoder" else DECODER_BLOCK_MAPPING
    match = re.fullmatch(r"layers\.(\d+)\.(.+)", rest)
    module = block_mapping[match.group(1)]
    rest = match.group(2)
    if module.startswith("res_unit"):
        match = re.fullmatch(r"layers\.(\d+)\.(.+)", rest)
        module = f"{module}.{RESIDUAL_UNIT_MAPPING[match.group(1)]}"
        rest = match.group(2)
    return f"{part}.block.{index - 1}.{module}.{rest}"


def convert_state_dict(original_state_dict: dict[str, torch.Tensor], num_blocks: int) -> dict[str, torch.Tensor]:
    """Renames the keys and folds each weight-normalized convolution's `weight_g` and `weight_v` into its weight."""
    state_dict = {}
    for key, value in original_state_dict.items():
        if key.endswith(".weight_v"):
            continue
        if key.endswith(".weight_g"):
            prefix = key[: -len(".weight_g")]
            # the computation of `torch.nn.utils.weight_norm` (dim 0), which the original runs at every forward
            value = torch._weight_norm(original_state_dict[f"{prefix}.weight_v"], value, 0)
            key = f"{prefix}.weight"
        state_dict[convert_key(key, num_blocks)] = value
    return state_dict


def convert_config(original_config: dict) -> YuE2AcousticTokenizerConfig:
    encoder_config, decoder_config = original_config["encoder_config"], original_config["decoder_config"]
    if (
        encoder_config["c_mults"] != decoder_config["c_mults"]
        or encoder_config["strides"] != decoder_config["strides"]
    ):
        raise ValueError("The converted model shares the channel multipliers and strides of the encoder and decoder.")
    if decoder_config.get("final_tanh", True) or not decoder_config.get("use_snake", False):
        raise ValueError("Only the released decoder (SnakeBeta activations, no final tanh) is supported.")
    return YuE2AcousticTokenizerConfig(
        sampling_rate=original_config["sample_rate"],
        audio_channels=original_config["audio_channels"],
        # the original's input convolution outputs `channels`, and each block `channels * c_mult`
        hidden_sizes=[decoder_config["channels"] * multiplier for multiplier in [1] + decoder_config["c_mults"]],
        downsampling_ratios=decoder_config["strides"],
        latent_dim=original_config["latent_dim"],
    )


def convert_checkpoint(checkpoint: str, output_dir: str, push_to_hub: str | None = None):
    checkpoint_dir = Path(checkpoint)
    if not checkpoint_dir.is_dir():
        checkpoint_dir = Path(snapshot_download(checkpoint, allow_patterns=["config.json", "*.safetensors"]))

    config = convert_config(json.loads((checkpoint_dir / "config.json").read_text()))
    state_dict = convert_state_dict(
        load_file(checkpoint_dir / "model.safetensors"), num_blocks=len(config.downsampling_ratios)
    )

    model = YuE2AcousticTokenizerModel(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.save_pretrained(output_dir)

    feature_extractor = EncodecFeatureExtractor(
        feature_size=config.audio_channels, sampling_rate=config.sampling_rate, padding_value=0.0
    )
    feature_extractor.save_pretrained(output_dir)

    # reload, as a user would
    YuE2AcousticTokenizerModel.from_pretrained(output_dir)
    print(f"Saved the converted model to {output_dir}")

    if push_to_hub:
        model.push_to_hub(push_to_hub)
        feature_extractor.push_to_hub(push_to_hub)
        print(f"Pushed the converted model to {push_to_hub}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="m-a-p/YuE2-Vae", help="Hub id or local directory of the original VAE."
    )
    parser.add_argument("--output_dir", required=True, help="Where to save the converted model.")
    parser.add_argument("--push_to_hub", default=None, help="Hub repository to push the converted model to.")
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.output_dir, args.push_to_hub)
