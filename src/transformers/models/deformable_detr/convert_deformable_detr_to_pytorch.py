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
"""Convert Deformable DETR checkpoints."""


import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import cached_download, hf_hub_url
from transformers import DeformableDetrConfig, DetrFeatureExtractor, DeformableDetrForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# here we list all keys to be renamed (original name on the left, our name on the right)
rename_keys = []
for i in range(6):
    # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
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

# query embeddings
rename_keys.extend(
    [
        ("query_embed.weight", "query_position_embeddings.weight"),
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


def read_in_q_k_v(state_dict):
    prefix = ""
    # transformer decoder self-attention layers
    for i in range(6):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_deformable_detr_checkpoint(checkpoint_path, single_scale, dilation, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our Deformable DETR structure.
    """

    # load default config
    config = DeformableDetrConfig()
    # set config attributes
    if single_scale:
        config.num_feature_levels = 1
    if dilation:
        config.dilation = True
    config.num_labels = 91
    repo_id = "datasets/huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename)), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    # load feature extractor
    feature_extractor = DetrFeatureExtractor(format="coco_detection")

    # prepare image
    img = prepare_img()
    encoding = feature_extractor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    logger.info(f"Converting model...")

    # load original state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")['model']
    # rename keys
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    state_dict = rename_backbone_keys(state_dict)
    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    prefix = "model."
    for key in state_dict.copy().keys():
        if not key.startswith("class_embed") and not key.startswith("bbox_embed"):
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
    # finally, create HuggingFace model and load state dict
    model = DeformableDetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()
    # verify our conversion
    outputs = model(pixel_values)

    expected_logits = torch.tensor([[ -9.6645,  -4.3449,  -5.8705],
        [ -9.7035,  -3.8504,  -5.0724],
        [-10.5634,  -5.3379,  -7.5116]])
    expected_boxes = torch.tensor([[0.8693, 0.2289, 0.2492],
        [0.3150, 0.5489, 0.5845],
        [0.5563, 0.7580, 0.8518]])
    assert torch.allclose(outputs.logits[0,:3,:3], expected_slice, atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0,:3,:3], expected_boxes, atol=1e-4)

    # Save model and feature extractor
    logger.info(f"Saving PyTorch model and feature extractor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", type=str, default="/home/niels/checkpoints/r50_deformable_detr-checkpoint.pth", help="Path to Pytorch checkpoint (.pth file) you'd like to convert."
    )
    parser.add_argument(
        "--single_scale", type=bool, default=False, help="Whether to set config.num_features_levels = 1."
    )
    parser.add_argument(
        "--dilation", type=bool, default=False, help="Whether to set config.dilation=True."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_deformable_detr_checkpoint(args.checkpoint_path, args.single_scale, args.dilation, args.pytorch_dump_folder_path)