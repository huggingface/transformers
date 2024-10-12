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
"""Convert DINOv2 + DPT checkpoints from the original repository. URL:
https://github.com/facebookresearch/dinov2/tree/main"""

import argparse
import itertools
import math
from pathlib import Path

import requests
import torch
from PIL import Image
from torchvision import transforms

from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation, DPTImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config(model_name):
    if "small" in model_name:
        # equivalent to stage 3, stage 6, stage 9, stage 12
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-small", out_indices=[3, 6, 9, 12], apply_layernorm=False, reshape_hidden_states=False
        )
        neck_hidden_sizes = [48, 96, 192, 384]
    elif "base" in model_name:
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-base", out_indices=[3, 6, 9, 12], apply_layernorm=False, reshape_hidden_states=False
        )
        neck_hidden_sizes = [96, 192, 384, 768]
    elif "large" in model_name:
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-large", out_indices=[5, 12, 18, 24], apply_layernorm=False, reshape_hidden_states=False
        )
        neck_hidden_sizes = [128, 256, 512, 1024]
    elif "giant" in model_name:
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-giant", out_indices=[10, 20, 30, 40], apply_layernorm=False, reshape_hidden_states=False
        )
        neck_hidden_sizes = [192, 384, 768, 1536]
    else:
        raise NotImplementedError("To do")

    config = DPTConfig(
        backbone_config=backbone_config,
        neck_hidden_sizes=neck_hidden_sizes,
        use_bias_in_fusion_residual=False,
        add_projection=True,
    )

    return config


# here we list all DPT keys to be renamed (original name on the left, our name on the right)
def create_rename_keys_dpt(config):
    rename_keys = []

    # fmt: off
    # activation postprocessing (projections, readout projections + resize blocks)
    for i in range(4):
        rename_keys.append((f"decode_head.reassemble_blocks.projects.{i}.conv.weight", f"neck.reassemble_stage.layers.{i}.projection.weight"))
        rename_keys.append((f"decode_head.reassemble_blocks.projects.{i}.conv.bias", f"neck.reassemble_stage.layers.{i}.projection.bias"))

        rename_keys.append((f"decode_head.reassemble_blocks.readout_projects.{i}.0.weight", f"neck.reassemble_stage.readout_projects.{i}.0.weight"))
        rename_keys.append((f"decode_head.reassemble_blocks.readout_projects.{i}.0.bias", f"neck.reassemble_stage.readout_projects.{i}.0.bias"))

        if i != 2:
            rename_keys.append((f"decode_head.reassemble_blocks.resize_layers.{i}.weight", f"neck.reassemble_stage.layers.{i}.resize.weight"))
            rename_keys.append((f"decode_head.reassemble_blocks.resize_layers.{i}.bias", f"neck.reassemble_stage.layers.{i}.resize.bias"))

    # fusion layers
    for i in range(4):
        rename_keys.append((f"decode_head.fusion_blocks.{i}.project.conv.weight", f"neck.fusion_stage.layers.{i}.projection.weight"))
        rename_keys.append((f"decode_head.fusion_blocks.{i}.project.conv.bias", f"neck.fusion_stage.layers.{i}.projection.bias"))
        if i != 0:
            rename_keys.append((f"decode_head.fusion_blocks.{i}.res_conv_unit1.conv1.conv.weight", f"neck.fusion_stage.layers.{i}.residual_layer1.convolution1.weight"))
            rename_keys.append((f"decode_head.fusion_blocks.{i}.res_conv_unit1.conv2.conv.weight", f"neck.fusion_stage.layers.{i}.residual_layer1.convolution2.weight"))
        rename_keys.append((f"decode_head.fusion_blocks.{i}.res_conv_unit2.conv1.conv.weight", f"neck.fusion_stage.layers.{i}.residual_layer2.convolution1.weight"))
        rename_keys.append((f"decode_head.fusion_blocks.{i}.res_conv_unit2.conv2.conv.weight", f"neck.fusion_stage.layers.{i}.residual_layer2.convolution2.weight"))

    # neck convolutions
    for i in range(4):
        rename_keys.append((f"decode_head.convs.{i}.conv.weight", f"neck.convs.{i}.weight"))

    # head
    rename_keys.append(("decode_head.project.conv.weight", "head.projection.weight"))
    rename_keys.append(("decode_head.project.conv.bias", "head.projection.bias"))

    for i in range(0, 5, 2):
        rename_keys.append((f"decode_head.conv_depth.head.{i}.weight", f"head.head.{i}.weight"))
        rename_keys.append((f"decode_head.conv_depth.head.{i}.bias", f"head.head.{i}.bias"))
    # fmt: on

    return rename_keys


# here we list all backbone keys to be renamed (original name on the left, our name on the right)
def create_rename_keys_backbone(config):
    rename_keys = []

    # fmt: off
    # patch embedding layer
    rename_keys.append(("cls_token", "backbone.embeddings.cls_token"))
    rename_keys.append(("mask_token", "backbone.embeddings.mask_token"))
    rename_keys.append(("pos_embed", "backbone.embeddings.position_embeddings"))
    rename_keys.append(("patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))

    # Transfomer encoder
    for i in range(config.backbone_config.num_hidden_layers):
        # layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"backbone.encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"backbone.encoder.layer.{i}.norm1.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"backbone.encoder.layer.{i}.norm2.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"backbone.encoder.layer.{i}.norm2.bias"))
        # MLP
        if config.backbone_config.use_swiglu_ffn:
            rename_keys.append((f"blocks.{i}.mlp.w12.weight", f"backbone.encoder.layer.{i}.mlp.w12.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w12.bias", f"backbone.encoder.layer.{i}.mlp.w12.bias"))
            rename_keys.append((f"blocks.{i}.mlp.w3.weight", f"backbone.encoder.layer.{i}.mlp.w3.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w3.bias", f"backbone.encoder.layer.{i}.mlp.w3.bias"))
        else:
            rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"backbone.encoder.layer.{i}.mlp.fc1.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"backbone.encoder.layer.{i}.mlp.fc1.bias"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"backbone.encoder.layer.{i}.mlp.fc2.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"backbone.encoder.layer.{i}.mlp.fc2.bias"))
        # layerscale
        rename_keys.append((f"blocks.{i}.ls1.gamma", f"backbone.encoder.layer.{i}.layer_scale1.lambda1"))
        rename_keys.append((f"blocks.{i}.ls2.gamma", f"backbone.encoder.layer.{i}.layer_scale2.lambda1"))
        # attention projection layer
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"backbone.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"backbone.encoder.layer.{i}.attention.output.dense.bias"))
    # fmt: on

    rename_keys.append(("norm.weight", "backbone.layernorm.weight"))
    rename_keys.append(("norm.bias", "backbone.layernorm.bias"))

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    for i in range(config.backbone_config.num_hidden_layers):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        hidden_size = config.backbone_config.hidden_size
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[:hidden_size]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            hidden_size : hidden_size * 2
        ]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-hidden_size:]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


name_to_url = {
    "dpt-dinov2-small-nyu": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_nyu_dpt_head.pth",
    "dpt-dinov2-small-kitti": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_kitti_dpt_head.pth",
    "dpt-dinov2-base-nyu": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_nyu_dpt_head.pth",
    "dpt-dinov2-base-kitti": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_kitti_dpt_head.pth",
    "dpt-dinov2-large-nyu": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_nyu_dpt_head.pth",
    "dpt-dinov2-large-kitti": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_kitti_dpt_head.pth",
    "dpt-dinov2-giant-nyu": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_nyu_dpt_head.pth",
    "dpt-dinov2-giant-kitti": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_kitti_dpt_head.pth",
}


def get_original_pixel_values(image):
    class CenterPadding:
        def __init__(self, multiple):
            super().__init__()
            self.multiple = multiple

        def _get_pad(self, size):
            new_size = math.ceil(size / self.multiple) * self.multiple
            pad_size = new_size - size
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
            return pad_size_left, pad_size_right

        def __call__(self, img):
            pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in img.shape[-2:][::-1]))
            output = torch.nn.functional.pad(img, pads)
            return output

        def __repr__(self):
            return self.__class__.__name__ + "()"

    def make_depth_transform() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                lambda x: 255.0 * x[:3],  # Discard alpha component and scale by 255
                transforms.Normalize(
                    mean=(123.675, 116.28, 103.53),
                    std=(58.395, 57.12, 57.375),
                ),
                CenterPadding(multiple=14),
            ]
        )

    transform = make_depth_transform()
    original_pixel_values = transform(image).unsqueeze(0)

    return original_pixel_values


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, verify_logits):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    # define DPT configuration based on URL
    checkpoint_url = name_to_url[model_name]
    config = get_dpt_config(model_name)

    # load original DPT state_dict from URL
    print("URL:", checkpoint_url)
    dpt_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["state_dict"]
    # rename keys
    rename_keys = create_rename_keys_dpt(config)
    for src, dest in rename_keys:
        rename_key(dpt_state_dict, src, dest)

    # load original backbone state_dict from URL
    if "small" in model_name:
        original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    elif "base" in model_name:
        original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    elif "large" in model_name:
        original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    elif "giant" in model_name:
        original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
    else:
        raise NotImplementedError("To do")
    original_model.eval()
    backbone_state_dict = original_model.state_dict()

    # rename keys
    rename_keys = create_rename_keys_backbone(config)
    for src, dest in rename_keys:
        rename_key(backbone_state_dict, src, dest)

    # read in qkv matrices
    read_in_q_k_v(backbone_state_dict, config)

    for key, val in backbone_state_dict.copy().items():
        val = backbone_state_dict.pop(key)
        if "w12" in key:
            key = key.replace("w12", "weights_in")
        if "w3" in key:
            key = key.replace("w3", "weights_out")
        backbone_state_dict[key] = val

    # merge state_dicts
    state_dict = {**backbone_state_dict, **dpt_state_dict}

    # load HuggingFace model
    model = DPTForDepthEstimation(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    assert missing_keys == [
        "neck.fusion_stage.layers.0.residual_layer1.convolution1.weight",
        "neck.fusion_stage.layers.0.residual_layer1.convolution2.weight",
    ]
    model.eval()

    # Verify image processor
    processor = DPTImageProcessor(
        do_resize=False,
        do_rescale=False,
        do_pad=True,
        size_divisor=14,
        do_normalize=True,
        image_mean=(123.675, 116.28, 103.53),
        image_std=(58.395, 57.12, 57.375),
    )

    image = prepare_img()
    pixel_values = processor(image, return_tensors="pt").pixel_values.float()
    original_pixel_values = get_original_pixel_values(image)

    assert torch.allclose(pixel_values, original_pixel_values)

    # Verify forward pass
    with torch.no_grad():
        outputs = model(pixel_values)

    predicted_depth = outputs.predicted_depth

    print("Shape of predicted depth:", predicted_depth.shape)
    print("First values of predicted depth:", predicted_depth[0, :3, :3])

    # assert logits
    if verify_logits:
        if model_name == "dpt-dinov2-small-nyu":
            expected_shape = torch.Size([1, 576, 736])
            expected_slice = torch.tensor(
                [[3.3576, 3.4741, 3.4345], [3.4324, 3.5012, 3.2775], [3.2560, 3.3563, 3.2354]]
            )

        assert predicted_depth.shape == torch.Size(expected_shape)
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=1e-5)
        print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model and processor to hub...")
        model.push_to_hub(repo_id=f"facebook/{model_name}")
        processor.push_to_hub(repo_id=f"facebook/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="dpt-dinov2-small-nyu",
        type=str,
        choices=name_to_url.keys(),
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
    parser.add_argument(
        "--verify_logits",
        action="store_true",
        required=False,
        help="Path to the output PyTorch model directory.",
    )

    args = parser.parse_args()
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits)
