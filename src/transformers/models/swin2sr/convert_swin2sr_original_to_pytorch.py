# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert Swin2SR checkpoints from the original repository. URL: https://github.com/mv-lab/swin2sr"""

import argparse

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import requests
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution


def get_config(model_name):
    config = Swin2SRConfig()

    if model_name == "swin2SR-classicalSR-x4-64":
        config.upscale = 4
    elif model_name == "swin2SR-lightweight-x2-64":
        config.depths = [6, 6, 6, 6]
        config.embed_dim = 60
        config.num_heads = [6, 6, 6, 6]
        config.upsampler = "pixelshuffledirect"
    elif model_name == "swin2SR-compressedSR-x4-48":
        config.image_size = 48
        config.upscale = 4
        config.upsampler = "pixelshuffle_aux"
    elif "real-sr" in model_name:
        config.upscale = 4
        config.upsampler = "nearest+conv"
    elif "real-sr-large" in model_name:
        config.upscale = 4
        config.upsampler = "nearest+conv"
        config.resi_connection = "3conv"
    elif "jpeg-car" in model_name:
        config.num_channels = 1
        config.upscale = 1
        config.image_size = 126
        config.window_size = 7
        config.img_range = 255.0
        config.upsampler = ""
    elif "color-jpeg-car" in model_name:
        config.upscale = 1
        config.image_size = 126
        config.window_size = 7
        config.img_range = 255.0
        config.upsampler = ""

    return config


def rename_key(name):
    if "patch_embed.proj" in name and "layers" not in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.patch_embeddings.layernorm")
    if "layers" in name:
        name = name.replace("layers", "encoder.stages")
    if "residual_group.blocks" in name:
        name = name.replace("residual_group.blocks", "layers")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "q_bias" in name:
        name = name.replace("q_bias", "query.bias")
    if "k_bias" in name:
        name = name.replace("k_bias", "key.bias")
    if "v_bias" in name:
        name = name.replace("v_bias", "value.bias")
    if "cpb_mlp" in name:
        name = name.replace("cpb_mlp", "continuous_position_bias_mlp")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "patch_embed.projection")

    if name == "norm.weight":
        name = "layernorm.weight"
    if name == "norm.bias":
        name = "layernorm.bias"

    if "upsample" in name or "conv_last" in name:
        pass
    else:
        name = "swin2sr." + name

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            stage_num = int(key_split[1])
            block_num = int(key_split[4])
            dim = config.embed_dim

            if "weight" in key:
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.key.weight"
                ] = val[dim : dim * 2, :]
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.query.bias"
                ] = val[:dim]
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.key.bias"
                ] = val[dim : dim * 2]
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.value.bias"
                ] = val[-dim:]
            pass
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_swin2sr_checkpoint(checkpoint_url, model_name, pytorch_dump_folder_path, push_to_hub):
    config = get_config(model_name)
    model = Swin2SRForImageSuperResolution(config)
    model.eval()

    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    new_state_dict = convert_state_dict(state_dict, config)
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    if len(missing_keys) > 0:
        raise ValueError("Missing keys when converting: {}".format(missing_keys))
    for key in unexpected_keys:
        if not ("relative_position_index" in key or "relative_coords_table" in key or "self_mask" in key):
            raise ValueError(f"Unexpected key {key} in state_dict")

    # TODO create feature extractor
    url = "https://github.com/mv-lab/swin2sr/blob/main/testsets/real-inputs/shanghai.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    transforms = Compose(
        [Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    pixel_values = transforms(image).unsqueeze(0)

    outputs = model(pixel_values)

    # assert values
    expected_shape = torch.Size([1, 3, 512, 512])
    expected_slice = torch.tensor(
        [[-0.7087, -0.7138, -0.6721], [-0.8340, -0.8095, -0.7298], [-0.9149, -0.8414, -0.7940]]
    )
    assert (
        outputs.logits.shape == expected_shape
    ), f"Shape of logits should be {expected_shape}, but is {outputs.logits.shape}"
    assert torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-3)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

        # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth",
        type=str,
        help="URL of the original Swin2SR checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--model_name",
        default="swin2SR-classical-sr-x2-64",
        type=str,
        help="Name of the Swin2SR model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the converted model to the hub.")

    args = parser.parse_args()
    convert_swin2sr_checkpoint(args.checkpoint_url, args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
