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
"""Convert RT Detr checkpoints with Timm backbone"""

import argparse
from pathlib import Path

import requests
import torch
from PIL import Image

from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_rt_detr_config(model_name: str) -> RTDetrConfig:
    config = RTDetrConfig()

    if model_name == "rtdetr_r18vd":
        pass
    elif model_name == "rtdetr_r34vd":
        pass
    elif model_name == "rtdetr_r50vd_m":
        pass
    elif model_name == "rtdetr_r50vd":
        pass
    elif model_name == "rtdetr_r101vd":
        pass
    elif model_name == "rtdetr_18vd_coco_o365":
        pass
    elif model_name == "rtdetr_r50vd_coco_o365":
        pass
    elif model_name == "rtdetr_r101vd_coco_o365":
        pass

    return config


def create_rename_keys(config):
    # here we list all keys to be renamed (original name on the left, our name on the right)
    rename_keys = []

    # stem
    # fmt: off
    last_key = ["weight", "bias", "running_mean", "running_var"]

    rename_keys.append(("backbone.conv1.conv1_1.conv.weight", "model.backbone.model.conv1.0.weight"))
    for last in last_key:
        rename_keys.append((f"backbone.conv1.conv1_1.norm.{last}", f"model.backbone.model.conv1.1.{last}"))

    rename_keys.append(("backbone.conv1.conv1_2.conv.weight", "model.backbone.model.conv1.3.weight"))
    for last in last_key:
        rename_keys.append((f"backbone.conv1.conv1_2.norm.{last}", f"model.backbone.model.conv1.3.{last}"))

    rename_keys.append(("backbone.conv1.conv1_3.conv.weight", "model.backbone.model.conv1.6.weight"))
    for last in last_key:
        rename_keys.append((f"backbone.conv1.conv1_3.norm.{last}", f"model.backbone.model.conv1.3.{last}"))

    # stages
    # TO DO : make this as function
    layer = [3,4,6,3]
    for stage_idx in range(4):
        for layer_idx in range(layer[stage_idx]):
            # shortcut
            if layer_idx == 0:
                if stage_idx == 0:
                    rename_keys.append(
                        (
                            f"backbone.res_layers.{stage_idx}.blocks.0.short.conv.weight",
                            f"model.backbone.model.layer{stage_idx+1}.0.downsample.1.weight",
                        )
                    )
                else:
                    rename_keys.append(
                        (
                            f"backbone.res_layers.{stage_idx}.blocks.0.short.conv.conv.weight",
                            f"model.backbone.model.layer{stage_idx+1}.0.downsample.1.weight",
                        )
                    )
            rename_keys.append(
                (
                    f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2a.conv.weight",
                    f"model.backbone.model.layer{stage_idx+1}.{layer_idx}.conv1.weight",
                )
            )
            for last in last_key:
                rename_keys.append((f"backbone.conv1.conv1_3.norm.{last}", f"model.backbone.model.conv1.3.{last}"))

            rename_keys.append(
                (
                    f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2b.conv.weight",
                    f"model.backbone.model.layer{stage_idx+1}.{layer_idx}.conv2.weight",
                )
            )
            rename_keys.append(
                (
                    f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2c.conv.weight",
                    f"model.backbone.model.layer{stage_idx+1}.{layer_idx}.conv3.weight",
                )
            )
    # fmt: on

    for i in range(config.encoder_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.self_attn.out_proj.weight",
                f"model.encoder.encoder.{i}.layers.0.self_attn.out_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.self_attn.out_proj.bias",
                f"model.encoder.encoder.{i}.layers.0.self_attn.out_proj.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear1.weight",
                f"model.encoder.encoder.{i}.layers.0.linear1.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear1.bias",
                f"model.encoder.encoder.{i}.layers.0.linear1.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear2.weight",
                f"model.encoder.encoder.{i}.layers.0.linear2.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear2.bias",
                f"model.encoder.encoder.{i}.layers.0.linear2.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm1.weight",
                f"model.encoder.encoder.{i}.layers.0.self_attn_layer_norm.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm1.bias",
                f"model.encoder.encoder.{i}.layers.0.self_attn_layer_norm.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm2.weight",
                f"model.encoder.encoder.{i}.layers.0.final_layer_norm.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm2.bias",
                f"model.encoder.encoder.{i}.layers.0.final_layer_norm.bias",
            )
        )

    for i in range(len(config.encoder_in_channels) - 1):
        # encoder layers: hybridencoder parts
        for j in range(1, 3):
            rename_keys.append(
                (f"encoder.fpn_blocks.{i}.conv{j}.conv.weight", f"model.encoder.fpn_blocks.{i}.conv{j}.conv.weight")
            )
            rename_keys.append(
                (f"encoder.fpn_blocks.{i}.conv{j}.norm.weight", f"model.encoder.fpn_blocks.{i}.conv{j}.norm.weight")
            )
            rename_keys.append(
                (f"encoder.fpn_blocks.{i}.conv{j}.norm.bias", f"model.encoder.fpn_blocks.{i}.conv{j}.norm.bias")
            )
        for j in range(3):
            for k in range(1, 3):
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.conv.weight",
                        f"model.encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.conv.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.norm.weight",
                        f"model.encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.norm.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.norm.bias",
                        f"model.encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.norm.bias",
                    )
                )

        for j in range(1, 3):
            rename_keys.append(
                (f"encoder.pan_blocks.{i}.conv{j}.conv.weight", f"model.encoder.pan_blocks.{i}.conv{j}.conv.weight")
            )
            rename_keys.append(
                (f"encoder.pan_blocks.{i}.conv{j}.norm.weight", f"model.encoder.pan_blocks.{i}.conv{j}.norm.weight")
            )
            rename_keys.append(
                (f"encoder.pan_blocks.{i}.conv{j}.norm.bias", f"model.encoder.pan_blocks.{i}.conv{j}.norm.bias")
            )
        for j in range(3):
            for k in range(1, 3):
                rename_keys.append(
                    (
                        f"encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.conv.weight",
                        f"model.encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.conv.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.norm.weight",
                        f"model.encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.norm.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.norm.bias",
                        f"model.encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.norm.bias",
                    )
                )

        rename_keys.append(
            (f"encoder.downsample_convs.{i}.conv.weight", f"model.encoder.downsample_convs.{i}.conv.weight")
        )
        rename_keys.append(
            (f"encoder.downsample_convs.{i}.norm.weight", f"model.encoder.downsample_convs.{i}.norm.weight")
        )
        rename_keys.append(
            (f"encoder.downsample_convs.{i}.norm.bias", f"model.encoder.downsample_convs.{i}.norm.bias")
        )

    for i in range(config.decoder_layers):
        # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.sampling_offsets.weight",
                f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.sampling_offsets.bias",
                f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.attention_weights.weight",
                f"model.decoder.layers.{i}.encoder_attn.attention_weights.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.attention_weights.bias",
                f"model.decoder.layers.{i}.encoder_attn.attention_weights.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.value_proj.weight",
                f"model.decoder.layers.{i}.encoder_attn.value_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.value_proj.bias",
                f"model.decoder.layers.{i}.encoder_attn.value_proj.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.output_proj.weight",
                f"model.decoder.layers.{i}.encoder_attn.output_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.output_proj.bias",
                f"model.decoder.layers.{i}.encoder_attn.output_proj.bias",
            )
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm1.weight", f"model.decoder.layers.{i}.self_attn_layer_norm.weight")
        )
        rename_keys.append((f"decoder.decoder.layers.{i}.norm1.bias", f"model.decoder.layers.{i}.self_attn_layer_norm.bias"))
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm2.weight", f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight")
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm2.bias", f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias")
        )
        rename_keys.append((f"decoder.decoder.layers.{i}.linear1.weight", f"model.decoder.layers.{i}.fc1.weight"))
        rename_keys.append((f"decoder.decoder.layers.{i}.linear1.bias", f"model.decoder.layers.{i}.fc1.bias"))
        rename_keys.append((f"decoder.decoder.layers.{i}.linear2.weight", f"model.decoder.layers.{i}.fc2.weight"))
        rename_keys.append((f"decoder.decoder.layers.{i}.linear2.bias", f"model.decoder.layers.{i}.fc2.bias"))
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm3.weight", f"model.decoder.layers.{i}.final_layer_norm.weight")
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm3.bias", f"model.decoder.layers.{i}.final_layer_norm.bias")
        )

    for i in range(config.decoder_layers):
        # decoder + class and bounding box heads
        rename_keys.append(
            (
                f"decoder.dec_score_head.{i}.weight",
                f"model.decoder.class_embed.{i}.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_score_head.{i}.bias",
                f"model.decoder.class_embed.{i}.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.0.weight",
                f"model.decoder.bbox_embed.{i}.layers.0.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.0.bias",
                f"model.decoder.bbox_embed.{i}.layers.0.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.1.weight",
                f"model.decoder.bbox_embed.{i}.layers.1.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.1.bias",
                f"model.decoder.bbox_embed.{i}.layers.1.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.2.weight",
                f"model.decoder.bbox_embed.{i}.layers.2.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.2.bias",
                f"model.decoder.bbox_embed.{i}.layers.2.bias",
            )
        )

    # decoder projection
    for i in range(len(config.decoder_in_channels)):
        rename_keys.append(
            (
                f"decoder.input_proj.{i}.conv.weight",
                f"model.decoder_input_proj.{i}.0.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.input_proj.{i}.norm.weight",
                f"model.decoder_input_proj.{i}.1.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.input_proj.{i}.norm.bias",
                f"model.decoder_input_proj.{i}.1.bias",
            )
        )

    # convolutional projection + query embeddings + layernorm of decoder + class and bounding box heads
    rename_keys.extend(
        [
            ("encoder.input_proj.0.0.weight", "model.encoder_input_proj.0.0.weight"),
            ("encoder.input_proj.0.1.weight", "model.encoder_input_proj.0.1.weight"),
            ("encoder.input_proj.0.1.bias", "model.encoder_input_proj.0.1.bias"),
            ("encoder.input_proj.1.0.weight", "model.encoder_input_proj.1.0.weight"),
            ("encoder.input_proj.1.1.weight", "model.encoder_input_proj.1.1.weight"),
            ("encoder.input_proj.1.1.bias", "model.encoder_input_proj.1.1.bias"),
            ("encoder.input_proj.2.0.weight", "model.encoder_input_proj.2.0.weight"),
            ("encoder.input_proj.2.1.weight", "model.encoder_input_proj.2.1.weight"),
            ("encoder.input_proj.2.1.bias", "model.encoder_input_proj.2.1.bias"),
            ("decoder.denoising_class_embed.weight", "model.denoising_class_embed.weight"),
            ("decoder.query_pos_head.layers.0.weight", "model.query_pos_head.layers.0.weight"),
            ("decoder.query_pos_head.layers.0.bias", "model.query_pos_head.layers.0.bias"),
            ("decoder.query_pos_head.layers.1.weight", "model.query_pos_head.layers.1.weight"),
            ("decoder.query_pos_head.layers.1.bias", "model.query_pos_head.layers.1.bias"),
            ("decoder.enc_output.0.weight", "model.enc_output.0.weight"),
            ("decoder.enc_output.0.bias", "model.enc_output.0.bias"),
            ("decoder.enc_output.1.weight", "model.enc_output.1.weight"),
            ("decoder.enc_output.1.bias", "model.enc_output.1.bias"),
            ("decoder.enc_score_head.weight", "model.enc_score_head.weight"),
            ("decoder.enc_score_head.bias", "model.enc_score_head.bias"),
            ("decoder.enc_bbox_head.layers.0.weight", "model.enc_bbox_head.layers.0.weight"),
            ("decoder.enc_bbox_head.layers.0.bias", "model.enc_bbox_head.layers.0.bias"),
            ("decoder.enc_bbox_head.layers.1.weight", "model.enc_bbox_head.layers.1.weight"),
            ("decoder.enc_bbox_head.layers.1.bias", "model.enc_bbox_head.layers.1.bias"),
            ("decoder.enc_bbox_head.layers.2.weight", "model.enc_bbox_head.layers.2.weight"),
            ("decoder.enc_bbox_head.layers.2.bias", "model.enc_bbox_head.layers.2.bias"),
        ]
    )

    return rename_keys


def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val


def read_in_q_k_v(state_dict, config):
    prefix = ""

    # first: transformer encoder
    for i in range(config.encoder_layers):
        # read in weights + bias of input projection layer (in PyTorch's MultiHeadAttention, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.bias"] = in_proj_bias[:256]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.bias"] = in_proj_bias[-256:]
    # next: transformer decoder (which is a bit more complex because it also includes cross-attention)
    for i in range(config.decoder_layers):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_rt_detr_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, repo_id):
    """
    Copy/paste/tweak model's weights to our RTDETR structure.
    """

    # load default config
    config = get_rt_detr_config(model_name)

    # load original model from torch hub
    model_name_to_checkpoint_url = {
        "rtdetr_r18vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
        "rtdetr_r34vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth",
        "rtdetr_r50vd_m": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth",
        "rtdetr_r50vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth",
        "rtdetr_r101vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth",
        "rtdetr_18vd_coco_o365": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth",
        "rtdetr_r50vd_coco_o365": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth",
        "rtdetr_r101vd_coco_o365": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth",
    }
    logger.info(f"Converting model {model_name}...")
    state_dict = torch.hub.load_state_dict_from_url(model_name_to_checkpoint_url[model_name], map_location="cpu")[
        "ema"
    ]["module"]

    # rename keys
    for src, dest in create_rename_keys(config):
        rename_key(state_dict, src, dest)
    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict, config)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    prefix = ""
    for key in state_dict.copy().keys():
        if not key.startswith("class_embed") and not key.startswith("bbox_embed"):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val

    # finally, create HuggingFace model and load state dict
    model = RTDetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    # load image processor
    image_processor = RTDetrImageProcessor()

    # prepare image
    img = prepare_img()
    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pixel_values = pixel_values.to(device)

    # Pass image by the model
    outputs = model(pixel_values)

    if model_name == "rtdetr_r18vd":
        pass
    elif model_name == "rtdetr_r34vd":
        pass
    elif model_name == "rtdetr_r50vd_m":
        pass
    elif model_name == "rtdetr_r50vd":
        expected_slice_logits = torch.tensor(
            [
                [-4.6476398, -5.001154, -4.9785104],
                [-4.1593494, -4.7038546, -5.946485],
                [-4.4374595, -4.658361, -6.2352347],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.16880608, 0.19992264, 0.21225442],
                [0.76837635, 0.4122631, 0.46368608],
                [0.2595386, 0.5483334, 0.4777486],
            ]
        )
    elif model_name == "rtdetr_r101vd":
        pass
    elif model_name == "rtdetr_18vd_coco_o365":
        pass
    elif model_name == "rtdetr_r50vd_coco_o365":
        pass
    elif model_name == "rtdetr_r101vd_coco_o365":
        pass
    else:
        raise ValueError(f"Unknown rt_detr_name: {model_name}")

    assert torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits, atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-3)

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # Upload model, image processor and config to the hub
        logger.info("Uploading PyTorch model and image processor to the hub...")
        config.push_to_hub(
            repo_id=repo_id, commit_message="Add config from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py"
        )
        model.push_to_hub(
            repo_id=repo_id, commit_message="Add model from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py"
        )
        image_processor.push_to_hub(
            repo_id=repo_id,
            commit_message="Add image processor from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="rtdetr_r50vd",
        type=str,
        help="model_name of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub or not.")
    parser.add_argument(
        "--repo_id",
        type=str,
        help="repo_id where the model will be pushed to.",
    )
    args = parser.parse_args()
    convert_rt_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)
