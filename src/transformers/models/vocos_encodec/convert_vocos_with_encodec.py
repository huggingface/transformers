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
import argparse
import io
import os

import torch

from transformers import (
    EncodecModel,
    VocosEncodecConfig,
    VocosEncodecModel,
    VocosEncodecProcessor,
    VocosFeatureExtractor,
)


def convert_old_keys_to_new_keys(original_state_dict: dict) -> dict:
    converted_checkpoint = {}
    for old_key, value in original_state_dict.items():
        if old_key == "feature_extractor.codebook_weights":
            new_key = old_key.replace("feature_extractor.codebook_weights", "codebook_weights")
            converted_checkpoint[new_key] = value
        elif old_key.startswith("backbone.") or old_key in ["norm.scale.weight", "norm.shift.weight"]:
            # Remove backbone prefix and flatten the structure
            new_key = old_key.replace("backbone.embed.", "embed.")
            new_key = new_key.replace("backbone.norm.", "norm.")
            new_key = new_key.replace("backbone.convnext.", "layers.")
            new_key = new_key.replace("backbone.final_layer_norm.", "final_layer_norm.")
            new_key = new_key.replace(".gamma", ".layer_scale_parameter")
            # Handle adaptive layer norm mappings
            new_key = new_key.replace(".norm.scale.weight", ".norm.weight")
            new_key = new_key.replace(".norm.shift.weight", ".norm.bias")
            # Handle top-level adaptive layer norm mappings
            if new_key == "norm.scale.weight":
                new_key = "norm.weight"
            elif new_key == "norm.shift.weight":
                new_key = "norm.bias"
            converted_checkpoint[new_key] = value
        elif old_key.startswith("head."):
            # Rename ISTFT head to decoder
            new_key = old_key.replace("head.", "decoder.")
            if "istft.window" in new_key:
                new_key = new_key.replace("istft.window", "window")
            converted_checkpoint[new_key] = value

    return converted_checkpoint


def safe_load(path: str) -> dict[str, torch.Tensor]:
    """
    Load only the tensor objects from a checkpoint, skipping any BytesIO
    """
    shard = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v for k, v in shard.items() if not isinstance(v, io.BytesIO)}


@torch.no_grad()
def convert_checkpoint(checkpoint_path, pytorch_dump_folder_path, push_to_hub=None):
    # determine shape of codebook weights
    # original: https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/feature_extractors.py#L74
    bandwidths = [1.5, 3.0, 6.0, 12.0]
    hf_encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").eval()
    num_quantizers = hf_encodec.quantizer.get_num_quantizers_for_bandwidth(bandwidth=max(bandwidths))
    codebook_weights = torch.cat(
        [layer.codebook.embed for layer in hf_encodec.quantizer.layers[:num_quantizers]], dim=0
    )

    # create model
    config = VocosEncodecConfig(
        bandwidths=bandwidths,
        codebook_dim=codebook_weights.shape[1],
        num_quantizers=codebook_weights.shape[0],
    )
    with torch.device("meta"):
        model = VocosEncodecModel(config)

    original_state_dict = safe_load(checkpoint_path)

    new_state_dict = convert_old_keys_to_new_keys(original_state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False, assign=True)
    print("Checkpoint loaded successfully")

    if len(unexpected_keys) != 0:
        raise ValueError(f"Unexpected keys: {unexpected_keys}")

    if len(missing_keys) != 0:
        raise ValueError(f"missing keys found: {missing_keys}")

    os.makedirs(pytorch_dump_folder_path, exist_ok=True)

    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=False)

    feature_extractor = VocosFeatureExtractor()

    processor = VocosEncodecProcessor(feature_extractor=feature_extractor, audio_tokenizer=None)

    processor.audio_tokenizer = hf_encodec

    processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        model.push_to_hub(push_to_hub)
        processor.push_to_hub(push_to_hub)
        print(f"Pushed model and processor to {push_to_hub}")


"""
# Download the original pytorch_model.bin
wget https://huggingface.co/charactr/vocos-encodec-24khz/resolve/main/pytorch_model.bin -O vocos_encodec_24khz_original.bin


# run conversion:
mkdir -p vocos-encodec-converted

python src/transformers/models/vocos/convert_vocos_with_encodec.py\
    --checkpoint_path vocos_encodec_24khz_original.bin\
    --pytorch_dump_folder_path vocos-encodec-converted \
    --push_to_hub hf-audio/vocos-encodec-24khz


# reload back
model = VocosModel.from_pretrained("vocos-encodec-converted")
processor  = VocosProcessor.from_pretrained("vocos-encodec-converted")
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
