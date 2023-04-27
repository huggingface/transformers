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
from pathlib import Path

import requests
import torch
from huggingface_hub import cached_download, hf_hub_url
from PIL import Image

from transformers import DeformableDetrConfig, DeformableDetrFeatureExtractor, DeformableDetrForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_key(orig_key):
    if "backbone.0.body" in orig_key:
        orig_key = orig_key.replace("backbone.0.body", "backbone.conv_encoder.model")
    if "transformer" in orig_key:
        orig_key = orig_key.replace("transformer.", "")
    if "norm1" in orig_key:
        if "encoder" in orig_key:
            orig_key = orig_key.replace("norm1", "self_attn_layer_norm")
        else:
            orig_key = orig_key.replace("norm1", "encoder_attn_layer_norm")
    if "norm2" in orig_key:
        if "encoder" in orig_key:
            orig_key = orig_key.replace("norm2", "final_layer_norm")
        else:
            orig_key = orig_key.replace("norm2", "self_attn_layer_norm")
    if "norm3" in orig_key:
        orig_key = orig_key.replace("norm3", "final_layer_norm")
    if "linear1" in orig_key:
        orig_key = orig_key.replace("linear1", "fc1")
    if "linear2" in orig_key:
        orig_key = orig_key.replace("linear2", "fc2")
    if "query_embed" in orig_key:
        orig_key = orig_key.replace("query_embed", "query_position_embeddings")
    if "cross_attn" in orig_key:
        orig_key = orig_key.replace("cross_attn", "encoder_attn")

    return orig_key


def read_in_q_k_v(state_dict):
    # transformer decoder self-attention layers
    for i in range(6):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"decoder.layers.{i}.self_attn.in_proj_bias")
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
def convert_deformable_detr_checkpoint(
    checkpoint_path,
    single_scale,
    dilation,
    with_box_refine,
    two_stage,
    pytorch_dump_folder_path,
    push_to_hub,
):
    """
    Copy/paste/tweak model's weights to our Deformable DETR structure.
    """

    # load default config
    config = DeformableDetrConfig()
    # set config attributes
    if single_scale:
        config.num_feature_levels = 1
    config.dilation = dilation
    config.with_box_refine = with_box_refine
    config.two_stage = two_stage
    # set labels
    config.num_labels = 91
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    # load feature extractor
    feature_extractor = DeformableDetrFeatureExtractor(format="coco_detection")

    # prepare image
    img = prepare_img()
    encoding = feature_extractor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    logger.info("Converting model...")

    # load original state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # verify our conversion
    outputs = model(pixel_values.to(device))

    expected_logits = torch.tensor(
        [[-9.6645, -4.3449, -5.8705], [-9.7035, -3.8504, -5.0724], [-10.5634, -5.3379, -7.5116]]
    )
    expected_boxes = torch.tensor([[0.8693, 0.2289, 0.2492], [0.3150, 0.5489, 0.5845], [0.5563, 0.7580, 0.8518]])

    if single_scale:
        expected_logits = torch.tensor(
            [[-9.9051, -4.2541, -6.4852], [-9.6947, -4.0854, -6.8033], [-10.0665, -5.8470, -7.7003]]
        )
        expected_boxes = torch.tensor([[0.7292, 0.4991, 0.5532], [0.7959, 0.2426, 0.4236], [0.7582, 0.3518, 0.4451]])

    if single_scale and dilation:
        expected_logits = torch.tensor(
            [[-8.9652, -4.1074, -5.6635], [-9.0596, -4.9447, -6.6075], [-10.1178, -4.5275, -6.2671]]
        )
        expected_boxes = torch.tensor([[0.7665, 0.4130, 0.4769], [0.8364, 0.1841, 0.3391], [0.6261, 0.3895, 0.7978]])

    if with_box_refine:
        expected_logits = torch.tensor(
            [[-8.8895, -5.4187, -6.8153], [-8.4706, -6.1668, -7.6184], [-9.0042, -5.5359, -6.9141]]
        )
        expected_boxes = torch.tensor([[0.7828, 0.2208, 0.4323], [0.0892, 0.5996, 0.1319], [0.5524, 0.6389, 0.8914]])

    if with_box_refine and two_stage:
        expected_logits = torch.tensor(
            [[-6.7108, -4.3213, -6.3777], [-8.9014, -6.1799, -6.7240], [-6.9315, -4.4735, -6.2298]]
        )
        expected_boxes = torch.tensor([[0.2583, 0.5499, 0.4683], [0.7652, 0.9068, 0.4882], [0.5490, 0.2763, 0.0564]])

    print("Logits:", outputs.logits[0, :3, :3])

    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits.to(device), atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes.to(device), atol=1e-4)

    print("Everything ok!")

    # Save model and feature extractor
    logger.info(f"Saving PyTorch model and feature extractor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    # Push to hub
    if push_to_hub:
        model_name = "deformable-detr"
        model_name += "-single-scale" if single_scale else ""
        model_name += "-dc5" if dilation else ""
        model_name += "-with-box-refine" if with_box_refine else ""
        model_name += "-two-stage" if two_stage else ""
        print("Pushing model to hub...")
        model.push_to_hub(repo_path_or_name=model_name, organization="nielsr", commit_message="Add model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/niels/checkpoints/deformable_detr/r50_deformable_detr-checkpoint.pth",
        help="Path to Pytorch checkpoint (.pth file) you'd like to convert.",
    )
    parser.add_argument("--single_scale", action="store_true", help="Whether to set config.num_features_levels = 1.")
    parser.add_argument("--dilation", action="store_true", help="Whether to set config.dilation=True.")
    parser.add_argument("--with_box_refine", action="store_true", help="Whether to set config.with_box_refine=True.")
    parser.add_argument("--two_stage", action="store_true", help="Whether to set config.two_stage=True.")
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to output PyTorch model.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()
    convert_deformable_detr_checkpoint(
        args.checkpoint_path,
        args.single_scale,
        args.dilation,
        args.with_box_refine,
        args.two_stage,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
