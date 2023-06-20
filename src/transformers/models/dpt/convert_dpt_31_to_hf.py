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

from transformers import BeitConfig, DPTConfig, DPTForDepthEstimation, DPTImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config():
    # beit-large-512 uses [5, 11, 17, 23]
    backbone_config = BeitConfig(
        image_size=512,
        num_hidden_layers=24,
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        use_relative_position_bias=True,
        out_features=["stage6", "stage12", "stage18", "stage24"],
    )

    # TODO get rid of config.hidden_size, using config.backbone_config.hidden_size instead
    config = DPTConfig(backbone_config=backbone_config, hidden_size=1024, neck_hidden_sizes=[256, 512, 1024, 1024])

    return config


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []

    # fmt: off
    # stem
    rename_keys.append(("pretrained.model.cls_token", "backbone.embeddings.cls_token"))
    rename_keys.append(("pretrained.model.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("pretrained.model.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))

    # Transfomer encoder
    for i in range(config.backbone_config.num_hidden_layers):
        rename_keys.append((f"pretrained.model.blocks.{i}.gamma_1", f"backbone.encoder.layer.{i}.lambda_1"))
        rename_keys.append((f"pretrained.model.blocks.{i}.gamma_2", f"backbone.encoder.layer.{i}.lambda_2"))
        rename_keys.append((f"pretrained.model.blocks.{i}.norm1.weight", f"backbone.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.norm1.bias", f"backbone.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.norm2.weight", f"backbone.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.norm2.bias", f"backbone.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.mlp.fc1.weight", f"backbone.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.mlp.fc1.bias", f"backbone.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.mlp.fc2.weight", f"backbone.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.mlp.fc2.bias", f"backbone.encoder.layer.{i}.output.dense.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.attn.proj.weight", f"backbone.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.attn.proj.bias", f"backbone.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.attn.relative_position_bias_table", f"backbone.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_bias_table"))
        rename_keys.append((f"pretrained.model.blocks.{i}.attn.relative_position_index", f"backbone.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_index"))

    # activation postprocessing (readout projections + resize blocks)
    for i in range(4):
        rename_keys.append((f"pretrained.act_postprocess{i+1}.0.project.0.weight", f"neck.reassemble_stage.readout_projects.{i}.0.weight"))
        rename_keys.append((f"pretrained.act_postprocess{i+1}.0.project.0.bias", f"neck.reassemble_stage.readout_projects.{i}.0.bias"))

        rename_keys.append((f"pretrained.act_postprocess{i+1}.3.weight", f"neck.reassemble_stage.layers.{i}.projection.weight"))
        rename_keys.append((f"pretrained.act_postprocess{i+1}.3.bias", f"neck.reassemble_stage.layers.{i}.projection.bias"))

        if i != 2:
            rename_keys.append((f"pretrained.act_postprocess{i+1}.4.weight", f"neck.reassemble_stage.layers.{i}.resize.weight"))
            rename_keys.append((f"pretrained.act_postprocess{i+1}.4.bias", f"neck.reassemble_stage.layers.{i}.resize.bias"))

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
def read_in_q_k_v(state_dict, config):
    hidden_size = config.backbone_config.hidden_size
    for i in range(config.backbone_config.num_hidden_layers):
        # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"pretrained.model.blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"pretrained.model.blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"pretrained.model.blocks.{i}.attn.v_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = v_bias


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    name_to_url = {
        "dpt-beit-large-512": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt",
    }

    # define DPT configuration based on URL
    checkpoint_url = name_to_url[model_name]
    config = get_dpt_config()
    # load original state_dict from URL
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # remove certain keys
    remove_ignore_keys_(state_dict)
    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # read in qkv matrices
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    model = DPTForDepthEstimation(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == []
    assert unexpected_keys == ["pretrained.model.fc_norm.weight", "pretrained.model.fc_norm.bias"]
    model.eval()

    # Check outputs on an image
    size = 512
    processor = DPTImageProcessor(size={"height": size, "width": size})

    image = prepare_img()
    processor(image, return_tensors="pt")

    import requests
    from PIL import Image
    from torchvision import transforms

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    transforms = transforms.Compose(
        [
            transforms.Resize((512, 512)),
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
    expected_shape = torch.Size([1, 512, 512])
    expected_slice = torch.tensor(
        [[2804.6260, 2792.5708, 2812.9263], [2772.0288, 2780.1118, 2796.2529], [2748.1094, 2766.6558, 2766.9834]]
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
        model.push_to_hub(repo_id=f"nielsr/{model_name}")
        processor.push_to_hub(repo_id=f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="dpt-beit-large-512",
        type=str,
        choices=["dpt-beit-large-512"],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub after conversion.",
    )

    args = parser.parse_args()
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
