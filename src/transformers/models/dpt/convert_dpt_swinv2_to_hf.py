# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert DPT 3.1 checkpoints from the MiDaS repository. URL: https://github.com/isl-org/MiDaS"""

import argparse
from pathlib import Path

import requests
import torch
from PIL import Image

from transformers import DPTConfig, DPTForDepthEstimation, DPTImageProcessor, Swinv2Config
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config(model_name):
    if "tiny" in model_name:
        embed_dim = 96
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
        window_size = 16
        # note: for Swinv2-tiny authors used the window_size = 16 variant
        # as seen here: https://github.com/isl-org/MiDaS/blob/bdc4ed64c095e026dc0a2f17cabb14d58263decb/midas/backbones/swin2.py#L26
        pretrained_window_sizes = (0, 0, 0, 0)
    elif "base" in model_name:
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
        window_size = 24
        pretrained_window_sizes = (12, 12, 12, 6)
    elif "large" in model_name:
        embed_dim = 192
        depths = (2, 2, 18, 2)
        num_heads = (6, 12, 24, 48)
        window_size = 24
        pretrained_window_sizes = (12, 12, 12, 6)

    if "384" in model_name:
        image_size = 384
    elif "256" in model_name:
        image_size = 256
    else:
        raise ValueError("Model not supported, to do")

    backbone_config = Swinv2Config(
        image_size=image_size,
        embed_dim=embed_dim,
        depths=depths,
        window_size=window_size,
        pretrained_window_sizes=pretrained_window_sizes,
        num_heads=num_heads,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )

    if model_name == "dpt-swinv2-tiny-256":
        neck_hidden_sizes = [96, 192, 384, 768]
    elif model_name == "dpt-swinv2-base-384":
        neck_hidden_sizes = [128, 256, 512, 1024]
    elif model_name == "dpt-swinv2-large-384":
        neck_hidden_sizes = [192, 384, 768, 1536]

    config = DPTConfig(backbone_config=backbone_config, neck_hidden_sizes=neck_hidden_sizes)

    return config, image_size


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []

    # fmt: off
    # stem
    rename_keys.append(("pretrained.model.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("pretrained.model.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("pretrained.model.patch_embed.norm.weight", "backbone.embeddings.norm.weight"))
    rename_keys.append(("pretrained.model.patch_embed.norm.bias", "backbone.embeddings.norm.bias"))

    # transformer encoder
    for i in range(len(config.backbone_config.depths)):
        for j in range(config.backbone_config.depths[i]):
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.logit_scale", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.logit_scale"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.cpb_mlp.0.weight", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.continuous_position_bias_mlp.0.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.cpb_mlp.0.bias", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.continuous_position_bias_mlp.0.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.cpb_mlp.2.weight", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.continuous_position_bias_mlp.2.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.q_bias", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.query.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.v_bias", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.value.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.proj.weight", f"backbone.encoder.layers.{i}.blocks.{j}.attention.output.dense.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.proj.bias", f"backbone.encoder.layers.{i}.blocks.{j}.attention.output.dense.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.norm1.weight", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_before.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.norm1.bias", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_before.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.mlp.fc1.weight", f"backbone.encoder.layers.{i}.blocks.{j}.intermediate.dense.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.mlp.fc1.bias", f"backbone.encoder.layers.{i}.blocks.{j}.intermediate.dense.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.mlp.fc2.weight", f"backbone.encoder.layers.{i}.blocks.{j}.output.dense.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.mlp.fc2.bias", f"backbone.encoder.layers.{i}.blocks.{j}.output.dense.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.norm2.weight", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_after.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.norm2.bias", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_after.bias"))

        # downsample parameters
        if i in [0,1,2]:
            rename_keys.append((f"pretrained.model.layers.{i}.downsample.reduction.weight", f"backbone.encoder.layers.{i}.downsample.reduction.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.downsample.norm.weight", f"backbone.encoder.layers.{i}.downsample.norm.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.downsample.norm.bias", f"backbone.encoder.layers.{i}.downsample.norm.bias"))

    # note: non-Transformer backbones like Swinv2, LeViT et al don't require activation postprocessing (readout projections + resize blocks)

    # refinenet (tricky here)
    mapping = {1:3, 2:2, 3:1, 4:0}

    for i in range(1, 5):
        j = mapping[i]
        rename_keys.append((f"scratch.refinenet{i}.out_conv.weight", f"neck.fusion_stage.layers.{j}.projection.weight"))
        rename_keys.append((f"scratch.refinenet{i}.out_conv.bias", f"neck.fusion_stage.layers.{j}.projection.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.bias"))

    # scratch convolutions
    for i in range(4):
        rename_keys.append((f"scratch.layer{i+1}_rn.weight", f"neck.convs.{i}.weight"))

    # head
    for i in range(0, 5, 2):
        rename_keys.append((f"scratch.output_conv.{i}.weight", f"head.head.{i}.weight"))
        rename_keys.append((f"scratch.output_conv.{i}.bias", f"head.head.{i}.bias"))

    return rename_keys


def remove_ignore_keys_(state_dict):
    ignore_keys = ["pretrained.model.head.weight", "pretrained.model.head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, model):
    for i in range(len(config.backbone_config.depths)):
        for j in range(config.backbone_config.depths[i]):
            dim = model.backbone.encoder.layers[i].blocks[j].attention.self.all_head_size
            # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
            in_proj_weight = state_dict.pop(f"pretrained.model.layers.{i}.blocks.{j}.attn.qkv.weight")
            # next, add query, keys and values (in that order) to the state dict
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[
                dim : dim * 2, :
            ]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[
                -dim:, :
            ]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, verify_logits, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    name_to_url = {
        "dpt-swinv2-tiny-256": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt",
        "dpt-swinv2-base-384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt",
        "dpt-swinv2-large-384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt",
    }

    # define DPT configuration based on URL
    checkpoint_url = name_to_url[model_name]
    config, image_size = get_dpt_config(model_name)
    # load original state_dict from URL
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")

    # load HuggingFace model
    model = DPTForDepthEstimation(config)

    # remove certain keys
    remove_ignore_keys_(state_dict)
    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # read in qkv matrices
    read_in_q_k_v(state_dict, config, model)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    model.eval()

    # Check outputs on an image
    processor = DPTImageProcessor(size={"height": image_size, "width": image_size})

    image = prepare_img()
    processor(image, return_tensors="pt")

    if verify_logits:
        from torchvision import transforms

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        transforms = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        pixel_values = transforms(image).unsqueeze(0)

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values)

        predicted_depth = outputs.predicted_depth

        print("Shape of predicted depth:", predicted_depth.shape)
        print("First values of predicted depth:", predicted_depth[0, :3, :3])

        # assert logits
        if model_name == "dpt-swinv2-base-384":
            # OK, checked
            expected_shape = torch.Size([1, 384, 384])
            expected_slice = torch.tensor(
                [
                    [1998.5575, 1997.3887, 2009.2981],
                    [1952.8607, 1979.6488, 2001.0854],
                    [1953.7697, 1961.7711, 1968.8904],
                ],
            )
        elif model_name == "dpt-swinv2-tiny-256":
            # OK, checked
            expected_shape = torch.Size([1, 256, 256])
            expected_slice = torch.tensor(
                [[978.9163, 976.5215, 978.5349], [974.1859, 971.7249, 975.8046], [971.3419, 970.3118, 971.6830]],
            )
        elif model_name == "dpt-swinv2-large-384":
            # OK, checked
            expected_shape = torch.Size([1, 384, 384])
            expected_slice = torch.tensor(
                [
                    [1203.7206, 1200.1495, 1197.8234],
                    [1196.2484, 1183.5033, 1186.4640],
                    [1178.8131, 1182.3260, 1174.3975],
                ],
            )

        assert predicted_depth.shape == torch.Size(expected_shape)
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice)
        print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model and processor to hub...")
        model.push_to_hub(repo_id=f"Intel/{model_name}")
        processor.push_to_hub(repo_id=f"Intel/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="dpt-swinv2-base-384",
        type=str,
        choices=["dpt-swinv2-tiny-256", "dpt-swinv2-base-384", "dpt-swinv2-large-384"],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--verify_logits",
        action="store_true",
        help="Whether to verify logits after conversion.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub after conversion.",
    )

    args = parser.parse_args()
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.verify_logits, args.push_to_hub)
