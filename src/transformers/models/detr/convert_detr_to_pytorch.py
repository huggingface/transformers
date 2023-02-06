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
"""Convert DETR checkpoints with native (Transformers) backbone."""


import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import DetrConfig, DetrForObjectDetection, DetrForSegmentation, DetrImageProcessor, ResNetConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_detr_config(model_name):
    config = DetrConfig(use_timm_backbone=False)

    # set backbone attributes
    if "resnet50" in model_name:
        pass
    elif "resnet101" in model_name:
        config.backbone_config = ResNetConfig.from_pretrained("microsoft/resnet-101")
    else:
        raise ValueError("Model name should include either resnet50 or resnet101")

    # set label attributes
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

    return config, is_panoptic


def create_rename_keys(config):
    # here we list all keys to be renamed (original name on the left, our name on the right)
    rename_keys = []

    # stem
    # fmt: off
    rename_keys.append(("backbone.0.body.conv1.weight", "backbone.conv_encoder.model.embedder.embedder.convolution.weight"))
    rename_keys.append(("backbone.0.body.bn1.weight", "backbone.conv_encoder.model.embedder.embedder.normalization.weight"))
    rename_keys.append(("backbone.0.body.bn1.bias", "backbone.conv_encoder.model.embedder.embedder.normalization.bias"))
    rename_keys.append(("backbone.0.body.bn1.running_mean", "backbone.conv_encoder.model.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("backbone.0.body.bn1.running_var", "backbone.conv_encoder.model.embedder.embedder.normalization.running_var"))
    # stages
    for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            # shortcut
            if layer_idx == 0:
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.0.weight",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.1.weight",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.1.bias",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.bias",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.1.running_mean",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.running_mean",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.1.running_var",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.running_var",
                    )
                )
            # 3 convs
            for i in range(3):
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.conv{i+1}.weight",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn{i+1}.weight",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn{i+1}.bias",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.bias",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn{i+1}.running_mean",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.running_mean",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn{i+1}.running_var",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.running_var",
                    )
                )
    # fmt: on

    for i in range(config.encoder_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append(
            (
                f"transformer.encoder.layers.{i}.self_attn.out_proj.weight",
                f"encoder.layers.{i}.self_attn.out_proj.weight",
            )
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
        rename_keys.append(
            (f"transformer.encoder.layers.{i}.norm1.bias", f"encoder.layers.{i}.self_attn_layer_norm.bias")
        )
        rename_keys.append(
            (f"transformer.encoder.layers.{i}.norm2.weight", f"encoder.layers.{i}.final_layer_norm.weight")
        )
        rename_keys.append((f"transformer.encoder.layers.{i}.norm2.bias", f"encoder.layers.{i}.final_layer_norm.bias"))
        # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms
        rename_keys.append(
            (
                f"transformer.decoder.layers.{i}.self_attn.out_proj.weight",
                f"decoder.layers.{i}.self_attn.out_proj.weight",
            )
        )
        rename_keys.append(
            (f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"decoder.layers.{i}.self_attn.out_proj.bias")
        )
        rename_keys.append(
            (
                f"transformer.decoder.layers.{i}.multihead_attn.out_proj.weight",
                f"decoder.layers.{i}.encoder_attn.out_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"transformer.decoder.layers.{i}.multihead_attn.out_proj.bias",
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
        rename_keys.append(
            (f"transformer.decoder.layers.{i}.norm1.bias", f"decoder.layers.{i}.self_attn_layer_norm.bias")
        )
        rename_keys.append(
            (f"transformer.decoder.layers.{i}.norm2.weight", f"decoder.layers.{i}.encoder_attn_layer_norm.weight")
        )
        rename_keys.append(
            (f"transformer.decoder.layers.{i}.norm2.bias", f"decoder.layers.{i}.encoder_attn_layer_norm.bias")
        )
        rename_keys.append(
            (f"transformer.decoder.layers.{i}.norm3.weight", f"decoder.layers.{i}.final_layer_norm.weight")
        )
        rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"decoder.layers.{i}.final_layer_norm.bias"))

    # convolutional projection + query embeddings + layernorm of decoder + class and bounding box heads
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
        ]
    )

    return rename_keys


def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val


def read_in_q_k_v(state_dict, is_panoptic=False):
    prefix = ""
    if is_panoptic:
        prefix = "detr."

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
    # next: transformer decoder (which is a bit more complex because it also includes cross-attention)
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
        # read in weights + bias of input projection layer of cross-attention
        in_proj_weight_cross_attn = state_dict.pop(
            f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_weight"
        )
        in_proj_bias_cross_attn = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_bias")
        # next, add query, keys and values (in that order) of cross-attention to the state dict
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_detr_checkpoint(model_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our DETR structure.
    """

    # load default config
    config, is_panoptic = get_detr_config(model_name)

    # load original model from torch hub
    logger.info(f"Converting model {model_name}...")
    detr = torch.hub.load("facebookresearch/detr", model_name, pretrained=True).eval()
    state_dict = detr.state_dict()
    # rename keys
    for src, dest in create_rename_keys(config):
        if is_panoptic:
            src = "detr." + src
        rename_key(state_dict, src, dest)
    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict, is_panoptic=is_panoptic)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    prefix = "detr.model." if is_panoptic else "model."
    for key in state_dict.copy().keys():
        if is_panoptic:
            if (
                key.startswith("detr")
                and not key.startswith("class_labels_classifier")
                and not key.startswith("bbox_predictor")
            ):
                val = state_dict.pop(key)
                state_dict["detr.model" + key[4:]] = val
            elif "class_labels_classifier" in key or "bbox_predictor" in key:
                val = state_dict.pop(key)
                state_dict["detr." + key] = val
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
    model = DetrForSegmentation(config) if is_panoptic else DetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    # verify our conversion on an image
    format = "coco_panoptic" if is_panoptic else "coco_detection"
    processor = DetrImageProcessor(format=format)

    encoding = processor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    original_outputs = detr(pixel_values)
    outputs = model(pixel_values)

    print("Logits:", outputs.logits[0, :3, :3])
    print("Original logits:", original_outputs["pred_logits"][0, :3, :3])

    assert torch.allclose(outputs.logits, original_outputs["pred_logits"], atol=1e-3)
    assert torch.allclose(outputs.pred_boxes, original_outputs["pred_boxes"], atol=1e-3)
    if is_panoptic:
        assert torch.allclose(outputs.pred_masks, original_outputs["pred_masks"], atol=1e-4)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        # Save model and image processor
        logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", default="detr_resnet50", type=str, help="Name of the DETR model you'd like to convert."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path)
