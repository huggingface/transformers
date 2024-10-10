# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Convert DAB-DETR checkpoints."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    DABDETRConfig,
    DABDETRForObjectDetection,
    DABDETRImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# here we list all keys to be renamed (original name on the left, HF name on the right)
rename_keys = []
for i in range(6):
    # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms + activation function
    # output projection
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.weight", f"encoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.bias", f"encoder.layers.{i}.self_attn.out_proj.bias")
    )
    # FFN layer
    # FFN 1
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.weight", f"encoder.layers.{i}.fc1.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.bias", f"encoder.layers.{i}.fc1.bias"))
    # FFN 2
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.weight", f"encoder.layers.{i}.fc2.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.bias", f"encoder.layers.{i}.fc2.bias"))
    # normalization layers
    # nm1
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.norm1.weight", f"encoder.layers.{i}.self_attn_layer_norm.weight")
    )
    rename_keys.append((f"transformer.encoder.layers.{i}.norm1.bias", f"encoder.layers.{i}.self_attn_layer_norm.bias"))
    # nm2
    rename_keys.append((f"transformer.encoder.layers.{i}.norm2.weight", f"encoder.layers.{i}.final_layer_norm.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.norm2.bias", f"encoder.layers.{i}.final_layer_norm.bias"))
    # activation function weight
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.activation.weight", f"encoder.layers.{i}.activation_fn.weight")
    )
    #########################################################################################################################################
    # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms + activiation function weight
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.self_attn.out_proj.weight",
            f"decoder.layers.{i}.self_attn.output_projection.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.self_attn.out_proj.bias",
            f"decoder.layers.{i}.self_attn.output_projection.bias",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.cross_attn.out_proj.weight",
            f"decoder.layers.{i}.cross_attn.output_projection.weight",
        )
    )
    # activation function weight
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.activation.weight", f"decoder.layers.{i}.activation_fn.weight")
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.cross_attn.out_proj.bias",
            f"decoder.layers.{i}.cross_attn.output_projection.bias",
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
        (f"transformer.decoder.layers.{i}.norm2.weight", f"decoder.layers.{i}.cross_attn_layer_norm.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.bias", f"decoder.layers.{i}.cross_attn_layer_norm.bias")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.weight", f"decoder.layers.{i}.final_layer_norm.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"decoder.layers.{i}.final_layer_norm.bias"))

    # q, k, v projections in self/cross-attention in decoder for DAB-DETR
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.sa_qcontent_proj.weight",
            f"decoder.layers.{i}.self_attn_query_content_proj.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.sa_kcontent_proj.weight",
            f"decoder.layers.{i}.self_attn_key_content_proj.weight",
        )
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qpos_proj.weight", f"decoder.layers.{i}.self_attn_query_pos_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kpos_proj.weight", f"decoder.layers.{i}.self_attn_key_pos_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_v_proj.weight", f"decoder.layers.{i}.self_attn_value_proj.weight")
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.ca_qcontent_proj.weight",
            f"decoder.layers.{i}.cross_attn_query_content_proj.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.ca_kcontent_proj.weight",
            f"decoder.layers.{i}.cross_attn_key_content_proj.weight",
        )
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kpos_proj.weight", f"decoder.layers.{i}.cross_attn_key_pos_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_v_proj.weight", f"decoder.layers.{i}.cross_attn_value_proj.weight")
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.ca_qpos_sine_proj.weight",
            f"decoder.layers.{i}.cross_attn_query_pos_sine_proj.weight",
        )
    )

    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.sa_qcontent_proj.bias",
            f"decoder.layers.{i}.self_attn_query_content_proj.bias",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.sa_kcontent_proj.bias",
            f"decoder.layers.{i}.self_attn_key_content_proj.bias",
        )
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qpos_proj.bias", f"decoder.layers.{i}.self_attn_query_pos_proj.bias")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kpos_proj.bias", f"decoder.layers.{i}.self_attn_key_pos_proj.bias")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_v_proj.bias", f"decoder.layers.{i}.self_attn_value_proj.bias")
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.ca_qcontent_proj.bias",
            f"decoder.layers.{i}.cross_attn_query_content_proj.bias",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.ca_kcontent_proj.bias",
            f"decoder.layers.{i}.cross_attn_key_content_proj.bias",
        )
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kpos_proj.bias", f"decoder.layers.{i}.cross_attn_key_pos_proj.bias")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_v_proj.bias", f"decoder.layers.{i}.cross_attn_value_proj.bias")
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.ca_qpos_sine_proj.bias",
            f"decoder.layers.{i}.cross_attn_query_pos_sine_proj.bias",
        )
    )

# convolutional projection + query embeddings + layernorm of decoder + class and bounding box heads
# for dab-DETR, also convert reference point head and query scale MLP
rename_keys.extend(
    [
        ("input_proj.weight", "input_projection.weight"),
        ("input_proj.bias", "input_projection.bias"),
        ("refpoint_embed.weight", "query_refpoint_embeddings.weight"),
        ("class_embed.weight", "class_embed.weight"),
        ("class_embed.bias", "class_embed.bias"),
        ("bbox_embed.layers.0.weight", "bbox_predictor.layers.0.weight"),
        ("bbox_embed.layers.0.bias", "bbox_predictor.layers.0.bias"),
        ("bbox_embed.layers.1.weight", "bbox_predictor.layers.1.weight"),
        ("bbox_embed.layers.1.bias", "bbox_predictor.layers.1.bias"),
        ("bbox_embed.layers.2.weight", "bbox_predictor.layers.2.weight"),
        ("bbox_embed.layers.2.bias", "bbox_predictor.layers.2.bias"),
        ("transformer.encoder.query_scale.layers.0.weight", "encoder.query_scale.layers.0.weight"),
        ("transformer.encoder.query_scale.layers.0.bias", "encoder.query_scale.layers.0.bias"),
        ("transformer.encoder.query_scale.layers.1.weight", "encoder.query_scale.layers.1.weight"),
        ("transformer.encoder.query_scale.layers.1.bias", "encoder.query_scale.layers.1.bias"),
        ("transformer.decoder.bbox_embed.layers.0.weight", "decoder.bbox_embed.layers.0.weight"),
        ("transformer.decoder.bbox_embed.layers.0.bias", "decoder.bbox_embed.layers.0.bias"),
        ("transformer.decoder.bbox_embed.layers.1.weight", "decoder.bbox_embed.layers.1.weight"),
        ("transformer.decoder.bbox_embed.layers.1.bias", "decoder.bbox_embed.layers.1.bias"),
        ("transformer.decoder.bbox_embed.layers.2.weight", "decoder.bbox_embed.layers.2.weight"),
        ("transformer.decoder.bbox_embed.layers.2.bias", "decoder.bbox_embed.layers.2.bias"),
        ("transformer.decoder.norm.weight", "decoder.layernorm.weight"),
        ("transformer.decoder.norm.bias", "decoder.layernorm.bias"),
        ("transformer.decoder.ref_point_head.layers.0.weight", "decoder.ref_point_head.layers.0.weight"),
        ("transformer.decoder.ref_point_head.layers.0.bias", "decoder.ref_point_head.layers.0.bias"),
        ("transformer.decoder.ref_point_head.layers.1.weight", "decoder.ref_point_head.layers.1.weight"),
        ("transformer.decoder.ref_point_head.layers.1.bias", "decoder.ref_point_head.layers.1.bias"),
        ("transformer.decoder.ref_anchor_head.layers.0.weight", "decoder.ref_anchor_head.layers.0.weight"),
        ("transformer.decoder.ref_anchor_head.layers.0.bias", "decoder.ref_anchor_head.layers.0.bias"),
        ("transformer.decoder.ref_anchor_head.layers.1.weight", "decoder.ref_anchor_head.layers.1.weight"),
        ("transformer.decoder.ref_anchor_head.layers.1.bias", "decoder.ref_anchor_head.layers.1.bias"),
        ("transformer.decoder.query_scale.layers.0.weight", "decoder.query_scale.layers.0.weight"),
        ("transformer.decoder.query_scale.layers.0.bias", "decoder.query_scale.layers.0.bias"),
        ("transformer.decoder.query_scale.layers.1.weight", "decoder.query_scale.layers.1.weight"),
        ("transformer.decoder.query_scale.layers.1.bias", "decoder.query_scale.layers.1.bias"),
        ("transformer.decoder.layers.0.ca_qpos_proj.weight", "decoder.layers.0.cross_attn_query_pos_proj.weight"),
        ("transformer.decoder.layers.0.ca_qpos_proj.bias", "decoder.layers.0.cross_attn_query_pos_proj.bias"),
    ]
)


def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val


def rename_backbone_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if "backbone.0.body" in key:
            new_key = key.replace("backbone.0.body", "backbone.conv_encoder.model._backbone")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def read_in_q_k_v(state_dict):
    prefix = ""

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
def convert_dab_detr_checkpoint(model_name, pretrained_model_weights_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our DAB-DETR structure.
    """

    # load modified config. Why? After loading the default config, the backbone kwargs are already set.
    if "dc5" in model_name:
        config = DABDETRConfig(dilation=True)
    else:
        # load default config
        config = DABDETRConfig()

    # set other attributes
    if "dab-detr-resnet-50-dc5" == model_name:
        config.temperature_height = 10
        config.temperature_width = 10
    if "fixxy" in model_name:
        config.random_refpoint_xy = True
    if "pat3" in model_name:
        config.num_patterns = 3
        # only when the number of patterns (num_patterns parameter in config) are more than 0 like r50-pat3 or r50dc5-pat3
        rename_keys.extend([("transformer.patterns.weight", "patterns.weight")])

    config.num_labels = 91
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    # load image processor
    format = "coco_detection"
    image_processor = DABDETRImageProcessor(format=format)

    # prepare image
    img = prepare_img()
    encoding = image_processor(images=[img, img], return_tensors="pt")

    logger.info(f"Converting model {model_name}...")

    # load original model from torch hub
    state_dict = torch.load(pretrained_model_weights_path, map_location=torch.device("cpu"))["model"]
    # rename keys
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    state_dict = rename_backbone_keys(state_dict)
    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    prefix = "model."
    for key in state_dict.copy().keys():
        if not key.startswith("class_embed") and not key.startswith("bbox_predictor"):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val

    # Expected logits and pred_boxes results of each model
    if model_name == "dab-detr-resnet-50":
        expected_slice_logits = torch.tensor(
            [[-10.1765, -5.5243, -8.9324], [-9.8138, -5.6721, -7.5161], [-10.3054, -5.6081, -8.5931]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.3708, 0.3000, 0.2753], [0.5211, 0.6125, 0.9495], [0.2897, 0.6730, 0.5459]]
        )
        logits_atol = 3e-4
        boxes_atol = 1e-4
    elif model_name == "dab-detr-resnet-50-pat3":
        expected_slice_logits = torch.tensor(
            [[-10.1069, -6.7068, -8.5944], [-9.4003, -7.3787, -8.7304], [-9.5858, -6.1514, -8.4744]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.5834, 0.6162, 0.2534], [0.6670, 0.2703, 0.1468], [0.5532, 0.1936, 0.0411]]
        )
        logits_atol = 1e-4
        boxes_atol = 1e-4
    elif model_name == "dab-detr-resnet-50-dc5":
        expected_slice_logits = torch.tensor(
            [[-9.9054, -6.0638, -7.8630], [-9.9112, -5.2952, -7.8175], [-9.8720, -5.3681, -7.7094]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.4077, 0.3644, 0.2689], [0.4429, 0.6903, 0.8238], [0.5188, 0.7933, 0.9989]]
        )
        logits_atol = 3e-3
        boxes_atol = 1e-3
    elif model_name == "dab-detr-resnet-50-dc5-pat3":
        expected_slice_logits = torch.tensor(
            [[-11.2264, -5.4028, -8.9815], [-10.8721, -6.0637, -9.1898], [-10.8535, -6.8360, -9.4203]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.8532, 0.5143, 0.1799], [0.6903, 0.3749, 0.3506], [0.5275, 0.2726, 0.0535]]
        )
        logits_atol = 1e-4
        boxes_atol = 1e-4
    elif model_name == "dab-detr-resnet-50-dc5-fixxy":
        expected_slice_logits = torch.tensor(
            [[-9.9362, -5.8105, -8.4952], [-9.6947, -4.9066, -8.3175], [-8.6919, -3.6328, -8.8972]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.4420, 0.3688, 0.2510], [0.5112, 0.7156, 0.9774], [0.4985, 0.4967, 0.9990]]
        )
        logits_atol = 5e-4
        boxes_atol = 1e-3

    # finally, create HuggingFace model and load state dict
    model = DABDETRForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.push_to_hub(repo_id=model_name, commit_message="Add new model")
    model.eval()
    # verify our conversion
    outputs = model(**encoding)

    assert torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits, atol=logits_atol)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, atol=boxes_atol)
    # Save model and image processor
    logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="dab-detr-resnet-50",
        type=str,
        help="Name of the DAB_DETR model you'd like to convert.",
    )
    parser.add_argument(
        "--pretrained_model_weights_path",
        default="",
        type=str,
        help="The path of the original model weights like: Users/username/Desktop/checkpoint.pth",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default="DAB_DETR", type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_dab_detr_checkpoint(args.model_name, args.pretrained_model_weights_path, args.pytorch_dump_folder_path)
