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
"""Convert Conditional DETR checkpoints."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    ConditionalDetrConfig,
    ConditionalDetrForObjectDetection,
    ConditionalDetrForSegmentation,
    ConditionalDetrImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# here we list all keys to be renamed (original name on the left, our name on the right)
rename_keys = []
for i in range(6):
    # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.weight", f"encoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.bias", f"encoder.layers.{i}.self_attn.out_proj.bias")
    )
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.weight", f"encoder.layers.{i}.fc1.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.bias", f"encoder.layers.{i}.fc1.bias"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.weight", f"encoder.layers.{i}.fc2.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.bias", f"encoder.layers.{i}.fc2.bias"))
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.norm1.weight", f"encoder.layers.{i}.self_attn_layer_norm.weight")
    )
    rename_keys.append((f"transformer.encoder.layers.{i}.norm1.bias", f"encoder.layers.{i}.self_attn_layer_norm.bias"))
    rename_keys.append((f"transformer.encoder.layers.{i}.norm2.weight", f"encoder.layers.{i}.final_layer_norm.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.norm2.bias", f"encoder.layers.{i}.final_layer_norm.bias"))
    # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"decoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"decoder.layers.{i}.self_attn.out_proj.bias")
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.cross_attn.out_proj.weight",
            f"decoder.layers.{i}.encoder_attn.out_proj.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.cross_attn.out_proj.bias",
            f"decoder.layers.{i}.encoder_attn.out_proj.bias",
        )
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.weight", f"decoder.layers.{i}.fc1.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.bias", f"decoder.layers.{i}.fc1.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.weight", f"decoder.layers.{i}.fc2.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.bias", f"decoder.layers.{i}.fc2.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm1.weight", f"decoder.layers.{i}.self_attn_layer_norm.weight")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.norm1.bias", f"decoder.layers.{i}.self_attn_layer_norm.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.weight", f"decoder.layers.{i}.encoder_attn_layer_norm.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.bias", f"decoder.layers.{i}.encoder_attn_layer_norm.bias")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.weight", f"decoder.layers.{i}.final_layer_norm.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"decoder.layers.{i}.final_layer_norm.bias"))

    # q, k, v projections in self/cross-attention in decoder for conditional DETR
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qcontent_proj.weight", f"decoder.layers.{i}.sa_qcontent_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kcontent_proj.weight", f"decoder.layers.{i}.sa_kcontent_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qpos_proj.weight", f"decoder.layers.{i}.sa_qpos_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kpos_proj.weight", f"decoder.layers.{i}.sa_kpos_proj.weight")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_v_proj.weight", f"decoder.layers.{i}.sa_v_proj.weight"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qcontent_proj.weight", f"decoder.layers.{i}.ca_qcontent_proj.weight")
    )
    # rename_keys.append((f"transformer.decoder.layers.{i}.ca_qpos_proj.weight", f"decoder.layers.{i}.ca_qpos_proj.weight"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kcontent_proj.weight", f"decoder.layers.{i}.ca_kcontent_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kpos_proj.weight", f"decoder.layers.{i}.ca_kpos_proj.weight")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.ca_v_proj.weight", f"decoder.layers.{i}.ca_v_proj.weight"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qpos_sine_proj.weight", f"decoder.layers.{i}.ca_qpos_sine_proj.weight")
    )

    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qcontent_proj.bias", f"decoder.layers.{i}.sa_qcontent_proj.bias")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kcontent_proj.bias", f"decoder.layers.{i}.sa_kcontent_proj.bias")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_qpos_proj.bias", f"decoder.layers.{i}.sa_qpos_proj.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_kpos_proj.bias", f"decoder.layers.{i}.sa_kpos_proj.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_v_proj.bias", f"decoder.layers.{i}.sa_v_proj.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qcontent_proj.bias", f"decoder.layers.{i}.ca_qcontent_proj.bias")
    )
    # rename_keys.append((f"transformer.decoder.layers.{i}.ca_qpos_proj.bias", f"decoder.layers.{i}.ca_qpos_proj.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kcontent_proj.bias", f"decoder.layers.{i}.ca_kcontent_proj.bias")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.ca_kpos_proj.bias", f"decoder.layers.{i}.ca_kpos_proj.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.ca_v_proj.bias", f"decoder.layers.{i}.ca_v_proj.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qpos_sine_proj.bias", f"decoder.layers.{i}.ca_qpos_sine_proj.bias")
    )

# convolutional projection + query embeddings + layernorm of decoder + class and bounding box heads
# for conditional DETR, also convert reference point head and query scale MLP
rename_keys.extend(
    [
        ("input_proj.weight", "input_projection.weight"),
        ("input_proj.bias", "input_projection.bias"),
        ("query_embed.weight", "query_position_embeddings.weight"),
        ("transformer.decoder.norm.weight", "decoder.layernorm.weight"),
        ("transformer.decoder.norm.bias", "decoder.layernorm.bias"),
        ("class_embed.weight", "class_labels_classifier.weight"),
        ("class_embed.bias", "class_labels_classifier.bias"),
        ("bbox_embed.layers.0.weight", "bbox_predictor.layers.0.weight"),
        ("bbox_embed.layers.0.bias", "bbox_predictor.layers.0.bias"),
        ("bbox_embed.layers.1.weight", "bbox_predictor.layers.1.weight"),
        ("bbox_embed.layers.1.bias", "bbox_predictor.layers.1.bias"),
        ("bbox_embed.layers.2.weight", "bbox_predictor.layers.2.weight"),
        ("bbox_embed.layers.2.bias", "bbox_predictor.layers.2.bias"),
        ("transformer.decoder.ref_point_head.layers.0.weight", "decoder.ref_point_head.layers.0.weight"),
        ("transformer.decoder.ref_point_head.layers.0.bias", "decoder.ref_point_head.layers.0.bias"),
        ("transformer.decoder.ref_point_head.layers.1.weight", "decoder.ref_point_head.layers.1.weight"),
        ("transformer.decoder.ref_point_head.layers.1.bias", "decoder.ref_point_head.layers.1.bias"),
        ("transformer.decoder.query_scale.layers.0.weight", "decoder.query_scale.layers.0.weight"),
        ("transformer.decoder.query_scale.layers.0.bias", "decoder.query_scale.layers.0.bias"),
        ("transformer.decoder.query_scale.layers.1.weight", "decoder.query_scale.layers.1.weight"),
        ("transformer.decoder.query_scale.layers.1.bias", "decoder.query_scale.layers.1.bias"),
        ("transformer.decoder.layers.0.ca_qpos_proj.weight", "decoder.layers.0.ca_qpos_proj.weight"),
        ("transformer.decoder.layers.0.ca_qpos_proj.bias", "decoder.layers.0.ca_qpos_proj.bias"),
    ]
)


def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val


def rename_backbone_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if "backbone.0.body" in key:
            new_key = key.replace("backbone.0.body", "backbone.conv_encoder.model")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def read_in_q_k_v(state_dict, is_panoptic=False):
    prefix = ""
    if is_panoptic:
        prefix = "conditional_detr."

    # first: transformer encoder
    for i in range(6):
        # read in weights + bias of input projection layer (in PyTorch's MultiHeadAttention, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_conditional_detr_checkpoint(model_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our CONDITIONAL_DETR structure.
    """

    # load default config
    config = ConditionalDetrConfig()
    # set backbone and dilation attributes
    if "resnet101" in model_name:
        config.backbone = "resnet101"
    if "dc5" in model_name:
        config.dilation = True
    is_panoptic = "panoptic" in model_name
    if is_panoptic:
        config.num_labels = 250
    else:
        config.num_labels = 91
        repo_id = "huggingface/label-files"
        filename = "coco-detection-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    # load image processor
    format = "coco_panoptic" if is_panoptic else "coco_detection"
    image_processor = ConditionalDetrImageProcessor(format=format)

    # prepare image
    img = prepare_img()
    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    logger.info(f"Converting model {model_name}...")

    # load original model from torch hub
    conditional_detr = torch.hub.load("DeppMeng/ConditionalDETR", model_name, pretrained=True).eval()
    state_dict = conditional_detr.state_dict()
    # rename keys
    for src, dest in rename_keys:
        if is_panoptic:
            src = "conditional_detr." + src
        rename_key(state_dict, src, dest)
    state_dict = rename_backbone_keys(state_dict)
    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict, is_panoptic=is_panoptic)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    prefix = "conditional_detr.model." if is_panoptic else "model."
    for key in state_dict.copy():
        if is_panoptic:
            if (
                key.startswith("conditional_detr")
                and not key.startswith("class_labels_classifier")
                and not key.startswith("bbox_predictor")
            ):
                val = state_dict.pop(key)
                state_dict["conditional_detr.model" + key[4:]] = val
            elif "class_labels_classifier" in key or "bbox_predictor" in key:
                val = state_dict.pop(key)
                state_dict["conditional_detr." + key] = val
            elif key.startswith("bbox_attention") or key.startswith("mask_head"):
                continue
            else:
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
        else:
            if not key.startswith("class_labels_classifier") and not key.startswith("bbox_predictor"):
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
    # finally, create HuggingFace model and load state dict
    model = ConditionalDetrForSegmentation(config) if is_panoptic else ConditionalDetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()
    model.push_to_hub(repo_id=model_name, organization="DepuMeng", commit_message="Add model")
    # verify our conversion
    original_outputs = conditional_detr(pixel_values)
    outputs = model(pixel_values)
    assert torch.allclose(outputs.logits, original_outputs["pred_logits"], atol=1e-4)
    assert torch.allclose(outputs.pred_boxes, original_outputs["pred_boxes"], atol=1e-4)
    if is_panoptic:
        assert torch.allclose(outputs.pred_masks, original_outputs["pred_masks"], atol=1e-4)

    # Save model and image processor
    logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="conditional_detr_resnet50",
        type=str,
        help="Name of the CONDITIONAL_DETR model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_conditional_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path)
