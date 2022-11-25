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
"""Convert Swin Transformer + UperNet checkpoints from mmsegmentation.

URL: https://github.com/open-mmlab/mmsegmentation/tree/master/configs/swin

Update: there seems to be an incompatibility with this version, due to a new implementation of their downsampling
operation using nn.Unfold.

TODO we need to update the parameters as shown here:
https://github.com/open-mmlab/mmdetection/blob/31c84958f54287a8be2b99cbf87a6dcf12e57753/mmdet/models/utils/ckpt_convert.py#L96.
"""

import argparse

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import requests
from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_upernet_config(model_name):
    if "tiny" in model_name:
        embed_dim = 96
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
    elif "small" in model_name:
        embed_dim = 96
        depths = (2, 2, 18, 2)
        num_heads = (3, 6, 12, 24)

    backbone_config = SwinConfig(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        output_hidden_states_before_downsampling=True,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    config = UperNetConfig(backbone_config=backbone_config, num_labels=150)

    return config


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []

    # fmt: off
    # stem
    rename_keys.append(("backbone.patch_embed.projection.weight", "backbone.swin.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("backbone.patch_embed.projection.bias", "backbone.swin.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("backbone.patch_embed.norm.weight", "backbone.swin.embeddings.norm.weight"))
    rename_keys.append(("backbone.patch_embed.norm.bias", "backbone.swin.embeddings.norm.bias"))
    # stages
    for i in range(len(config.backbone_config.depths)):
        for j in range(config.backbone_config.depths[i]):
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm1.weight", f"backbone.swin.encoder.layers.{i}.blocks.{j}.layernorm_before.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm1.bias", f"backbone.swin.encoder.layers.{i}.blocks.{j}.layernorm_before.bias"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.relative_position_bias_table", f"backbone.swin.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_bias_table"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.relative_position_index", f"backbone.swin.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_index"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.proj.weight", f"backbone.swin.encoder.layers.{i}.blocks.{j}.attention.output.dense.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.proj.bias", f"backbone.swin.encoder.layers.{i}.blocks.{j}.attention.output.dense.bias"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm2.weight", f"backbone.swin.encoder.layers.{i}.blocks.{j}.layernorm_after.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm2.bias", f"backbone.swin.encoder.layers.{i}.blocks.{j}.layernorm_after.bias"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.0.0.weight", f"backbone.swin.encoder.layers.{i}.blocks.{j}.intermediate.dense.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.0.0.bias", f"backbone.swin.encoder.layers.{i}.blocks.{j}.intermediate.dense.bias"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.1.weight", f"backbone.swin.encoder.layers.{i}.blocks.{j}.output.dense.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.1.bias", f"backbone.swin.encoder.layers.{i}.blocks.{j}.output.dense.bias"))

        if i < 3:
            rename_keys.append((f"backbone.stages.{i}.downsample.reduction.weight", f"backbone.swin.encoder.layers.{i}.downsample.reduction.weight"))
            rename_keys.append((f"backbone.stages.{i}.downsample.norm.weight", f"backbone.swin.encoder.layers.{i}.downsample.norm.weight"))
            rename_keys.append((f"backbone.stages.{i}.downsample.norm.bias", f"backbone.swin.encoder.layers.{i}.downsample.norm.bias"))
        rename_keys.append((f"backbone.norm{i}.weight", f"backbone.hidden_states_norms.{i}.weight"))
        rename_keys.append((f"backbone.norm{i}.bias", f"backbone.hidden_states_norms.{i}.bias"))

    # decode head
    rename_keys.extend(
        [
            ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
            ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
            ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
            ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
        ]
    )
    # fmt: on

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, backbone_config):
    num_features = [int(backbone_config.embed_dim * 2**i) for i in range(len(backbone_config.depths))]
    for i in range(len(backbone_config.depths)):
        dim = num_features[i]
        for j in range(backbone_config.depths[i]):
            # fmt: off
            # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
            in_proj_weight = state_dict.pop(f"backbone.stages.{i}.blocks.{j}.attn.w_msa.qkv.weight")
            in_proj_bias = state_dict.pop(f"backbone.stages.{i}.blocks.{j}.attn.w_msa.qkv.bias")
            # next, add query, keys and values (in that order) to the state dict
            state_dict[f"backbone.swin.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"backbone.swin.encoder.layers.{i}.blocks.{j}.attention.self.query.bias"] = in_proj_bias[: dim]
            state_dict[f"backbone.swin.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[
                dim : dim * 2, :
            ]
            state_dict[f"backbone.swin.encoder.layers.{i}.blocks.{j}.attention.self.key.bias"] = in_proj_bias[
                dim : dim * 2
            ]
            state_dict[f"backbone.swin.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[
                -dim :, :
            ]
            state_dict[f"backbone.swin.encoder.layers.{i}.blocks.{j}.attention.self.value.bias"] = in_proj_bias[-dim :]
            # fmt: on


image_transforms = Compose(
    [Resize((512, 512)), ToTensor(), Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)]
)


def correct_unfold_reduction_order(x):
    out_channel, in_channel = x.shape
    x = x.reshape(out_channel, 4, in_channel // 4)
    x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
    return x


def reverse_correct_unfold_reduction_order(x):
    out_channel, in_channel = x.shape
    x = x.reshape(out_channel, in_channel // 4, 4)
    x = x[:, :, [0, 2, 1, 3]].transpose(1, 2).reshape(out_channel, in_channel)

    return x


def correct_unfold_norm_order(x):
    in_channel = x.shape[0]
    x = x.reshape(4, in_channel // 4)
    x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
    return x


def reverse_correct_unfold_norm_order(x):
    in_channel = x.shape[0]
    x = x.reshape(in_channel // 4, 4)
    x = x[:, [0, 2, 1, 3]].transpose(0, 1).reshape(in_channel)
    return x


def convert_upernet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    model_name_to_url = {
        "upernet-swin-tiny-mmsegmentation": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth",
        "upernet-swin-small-mmsegmentation": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth",
    }
    checkpoint_url = model_name_to_url[model_name]
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", file_name=model_name)["state_dict"]

    for name, param in state_dict.items():
        print(name, param.shape)

    config = get_upernet_config(model_name)
    model = UperNetForSemanticSegmentation(config)
    model.eval()

    # replace "bn" => "batch_norm"
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if "bn" in key:
            key = key.replace("bn", "batch_norm")
        state_dict[key] = val

    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config.backbone_config)

    # fix downsample parameters
    for key, value in state_dict.items():
        if "downsample" in key:
            if "reduction" in key:
                state_dict[key] = reverse_correct_unfold_reduction_order(value)
            if "norm" in key:
                state_dict[key] = reverse_correct_unfold_norm_order(value)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == ["backbone.swin.layernorm.weight", "backbone.swin.layernorm.bias"]
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

    # verify on image
    url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    pixel_values = image_transforms(image).unsqueeze(0)

    print("Sum of pixel values:", pixel_values.sum().item())
    print("Mean of pixel values:", pixel_values.mean().item())

    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits

    print(logits.shape)
    print("First values of logits:", logits[0,0,:3,:3])
    # TODO assert values
    # expected_slice = torch.tensor(
    #     [[-8.8110, -7.5399, -7.5429], [-8.5200, -7.0736, -7.2054], [-8.5220, -7.2897, -7.3901]]
    # )
    # print("Logits:", outputs.logits[0, 0, :3, :3])
    # assert torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4)
    # print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and feature extractor for {model_name} to hub")
        model.push_to_hub(f"nielsr/{model_name}")
        # feature_extractor.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="upernet-swin-tiny-mmsegmentation",
        type=str,
        choices=["upernet-swin-tiny-mmsegmentation", "upernet-swin-small-mmsegmentation"],
        help="Name of the Swin + UperNet model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_upernet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
