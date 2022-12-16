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
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution, Swin2SRImageProcessor


def get_config(checkpoint_url):
    config = Swin2SRConfig()

    if "Swin2SR_ClassicalSR_X4_64" in checkpoint_url:
        config.upscale = 4
    elif "Swin2SR_CompressedSR_X4_48" in checkpoint_url:
        config.upscale = 4
        config.image_size = 48
        config.upsampler = "pixelshuffle_aux"
    elif "Swin2SR_Lightweight_X2_64" in checkpoint_url:
        config.depths = [6, 6, 6, 6]
        config.embed_dim = 60
        config.num_heads = [6, 6, 6, 6]
        config.upsampler = "pixelshuffledirect"
    elif "Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR" in checkpoint_url:
        config.upscale = 4
        config.upsampler = "nearest+conv"
    elif "Swin2SR_Jpeg_dynamic" in checkpoint_url:
        config.num_channels = 1
        config.upscale = 1
        config.image_size = 126
        config.window_size = 7
        config.img_range = 255.0
        config.upsampler = ""

    return config


def rename_key(name, config):
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

    if "conv_first" in name:
        name = name.replace("conv_first", "first_convolution")

    if (
        "upsample" in name
        or "conv_before_upsample" in name
        or "conv_bicubic" in name
        or "conv_up" in name
        or "conv_hr" in name
        or "conv_last" in name
        or "aux" in name
    ):
        # heads
        if "conv_last" in name:
            name = name.replace("conv_last", "final_convolution")
        if config.upsampler in ["pixelshuffle", "pixelshuffle_aux", "nearest+conv"]:
            if "conv_before_upsample.0" in name:
                name = name.replace("conv_before_upsample.0", "conv_before_upsample")
            if "upsample.0" in name:
                name = name.replace("upsample.0", "upsample.convolution_0")
            if "upsample.2" in name:
                name = name.replace("upsample.2", "upsample.convolution_1")
            name = "upsample." + name
        elif config.upsampler == "pixelshuffledirect":
            name = name.replace("upsample.0.weight", "upsample.conv.weight")
            name = name.replace("upsample.0.bias", "upsample.conv.bias")
        else:
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
            orig_state_dict[rename_key(key, config)] = val

    return orig_state_dict


def convert_swin2sr_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub):
    config = get_config(checkpoint_url)
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

    # verify values
    url = "https://github.com/mv-lab/swin2sr/blob/main/testsets/real-inputs/shanghai.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    processor = Swin2SRImageProcessor()
    # pixel_values = processor(image, return_tensors="pt").pixel_values

    image_size = 126 if "Jpeg" in checkpoint_url else 256
    transforms = Compose(
        [
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    pixel_values = transforms(image).unsqueeze(0)

    if config.num_channels == 1:
        pixel_values = pixel_values[:, 0, :, :].unsqueeze(1)

    outputs = model(pixel_values)

    # assert values
    if "Swin2SR_ClassicalSR_X2_64" in checkpoint_url:
        expected_shape = torch.Size([1, 3, 512, 512])
        expected_slice = torch.tensor(
            [[-0.7087, -0.7138, -0.6721], [-0.8340, -0.8095, -0.7298], [-0.9149, -0.8414, -0.7940]]
        )
    elif "Swin2SR_ClassicalSR_X4_64" in checkpoint_url:
        expected_shape = torch.Size([1, 3, 1024, 1024])
        expected_slice = torch.tensor(
            [[-0.7775, -0.8105, -0.8933], [-0.7764, -0.8356, -0.9225], [-0.7976, -0.8686, -0.9579]]
        )
    elif "Swin2SR_CompressedSR_X4_48" in checkpoint_url:
        # TODO values didn't match exactly here
        expected_shape = torch.Size([1, 3, 1024, 1024])
        expected_slice = torch.tensor(
            [[-0.8035, -0.7504, -0.7491], [-0.8538, -0.8124, -0.7782], [-0.8804, -0.8651, -0.8493]]
        )
    elif "Swin2SR_Lightweight_X2_64" in checkpoint_url:
        expected_shape = torch.Size([1, 3, 512, 512])
        expected_slice = torch.tensor(
            [[-0.7669, -0.8662, -0.8767], [-0.8810, -0.9962, -0.9820], [-0.9340, -1.0322, -1.1149]]
        )
    elif "Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR" in checkpoint_url:
        expected_shape = torch.Size([1, 3, 1024, 1024])
        expected_slice = torch.tensor(
            [[-0.5238, -0.5557, -0.6321], [-0.6016, -0.5903, -0.6391], [-0.6244, -0.6334, -0.6889]]
        )

    assert (
        outputs.reconstruction.shape == expected_shape
    ), f"Shape of reconstruction should be {expected_shape}, but is {outputs.reconstruction.shape}"
    assert torch.allclose(outputs.reconstruction[0, 0, :3, :3], expected_slice, atol=1e-3)
    print("Looks ok!")

    url_to_name = {
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth": (
            "swin2SR-classical-sr-x2-64"
        ),
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X4_64.pth": (
            "swin2SR-classical-sr-x4-64"
        ),
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_CompressedSR_X4_48.pth": (
            "swin2SR-compressed-sr-x4-48"
        ),
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_Lightweight_X2_64.pth": (
            "swin2SR-lightweight-x2-64"
        ),
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth": (
            "swin2SR-realworld-sr-x4-64-bsrgan-psnr"
        ),
    }
    model_name = url_to_name[checkpoint_url]

    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub(f"caidas/{model_name}")
        processor.push_to_hub(f"caidas/{model_name}")


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
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the converted model to the hub.")

    args = parser.parse_args()
    convert_swin2sr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
