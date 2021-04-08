# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert DETR checkpoints."""


import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from packaging import version
from PIL import Image

import requests
from transformers import DetrConfig, DetrForObjectDetection, DetrModel
from transformers.utils import logging
from transformers.utils.coco_classes import id2label


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# here we list all keys to be renamed (original name on the left, our name on the right)
rename_keys = []
for i in range(6):
    # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
    rename_keys.append(
        (
            f"transformer.encoder.layers.{i}.self_attn.out_proj.weight",
            f"encoder.layers.{i}.self_attn.out_proj.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.encoder.layers.{i}.self_attn.out_proj.bias",
            f"encoder.layers.{i}.self_attn.out_proj.bias",
        )
    )
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.weight", f"encoder.layers.{i}.fc1.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.bias", f"encoder.layers.{i}.fc1.bias"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.weight", f"encoder.layers.{i}.fc2.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.bias", f"encoder.layers.{i}.fc2.bias"))
    rename_keys.append(
        (
            f"transformer.encoder.layers.{i}.norm1.weight",
            f"encoder.layers.{i}.self_attn_layer_norm.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.encoder.layers.{i}.norm1.bias",
            f"encoder.layers.{i}.self_attn_layer_norm.bias",
        )
    )
    rename_keys.append(
        (
            f"transformer.encoder.layers.{i}.norm2.weight",
            f"encoder.layers.{i}.final_layer_norm.weight",
        )
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
        (
            f"transformer.decoder.layers.{i}.self_attn.out_proj.bias",
            f"decoder.layers.{i}.self_attn.out_proj.bias",
        )
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
        (
            f"transformer.decoder.layers.{i}.norm1.weight",
            f"decoder.layers.{i}.self_attn_layer_norm.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.norm1.bias",
            f"decoder.layers.{i}.self_attn_layer_norm.bias",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.norm2.weight",
            f"decoder.layers.{i}.encoder_attn_layer_norm.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.norm2.bias",
            f"decoder.layers.{i}.encoder_attn_layer_norm.bias",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.norm3.weight",
            f"decoder.layers.{i}.final_layer_norm.weight",
        )
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"decoder.layers.{i}.final_layer_norm.bias"))


# convolutional projection + query embeddings + layernorm of decoder
rename_keys.extend(
    [
        ("input_proj.weight", "input_projection.weight"),
        ("input_proj.bias", "input_projection.bias"),
        ("query_embed.weight", "query_position_embeddings.weight"),
        ("transformer.decoder.norm.weight", "decoder.layernorm.weight"),
        ("transformer.decoder.norm.bias", "decoder.layernorm.bias"),
    ]
)


def remove_object_detection_heads_(state_dict):
    ignore_keys = [
        "class_embed.weight",
        "class_embed.bias",
        "bbox_embed.layers.0.weight",
        "bbox_embed.layers.0.bias",
        "bbox_embed.layers.1.weight",
        "bbox_embed.layers.1.bias",
        "bbox_embed.layers.2.weight",
        "bbox_embed.layers.2.bias",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val


def read_in_q_k_v(state_dict):
    # first: transformer encoder
    for i in range(6):
        # read in weights + bias of input projection layer (in PyTorch's MultiHeadAttention, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"transformer.encoder.layers.{i}.self_attn.in_proj_bias")
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
        in_proj_weight = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
        # read in weights + bias of input projection layer of cross-attention
        in_proj_weight_cross_attn = state_dict.pop(f"transformer.decoder.layers.{i}.multihead_attn.in_proj_weight")
        in_proj_bias_cross_attn = state_dict.pop(f"transformer.decoder.layers.{i}.multihead_attn.in_proj_bias")
        # next, add query, keys and values (in that order) of cross-attention to the state dict
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]


# since we renamed the classification heads of the object detection model, we need to rename the original keys:
rename_keys_object_detection_model = [
    ("class_embed.weight", "class_labels_classifier.weight"),
    ("class_embed.bias", "class_labels_classifier.bias"),
    ("bbox_embed.layers.0.weight", "bbox_predictor.layers.0.weight"),
    ("bbox_embed.layers.0.bias", "bbox_predictor.layers.0.bias"),
    ("bbox_embed.layers.1.weight", "bbox_predictor.layers.1.weight"),
    ("bbox_embed.layers.1.bias", "bbox_predictor.layers.1.bias"),
    ("bbox_embed.layers.2.weight", "bbox_predictor.layers.2.weight"),
    ("bbox_embed.layers.2.bias", "bbox_predictor.layers.2.bias"),
]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    return img


@torch.no_grad()
def convert_detr_checkpoint(task, backbone, dilation, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our DETR structure.
    """

    config = DetrConfig()
    img = prepare_img()

    logger.info(f"Converting model for task {task}, with a {backbone} backbone, dilation set to {dilation}...")

    if task == "base_model":
        # load model from torch hub
        detr = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True).eval()
        state_dict = detr.state_dict()
        # rename keys
        for src, dest in rename_keys:
            rename_key(state_dict, src, dest)
        # query, key and value matrices need special treatment
        read_in_q_k_v(state_dict)
        # remove classification heads
        remove_object_detection_heads_(state_dict)
        # finally, create model and load state dict
        model = DetrModel(config).eval()
        model.load_state_dict(state_dict)
        # verify our conversion on the image
        outputs = model(img)
        assert outputs.last_hidden_state.shape == (1, config.num_queries, config.d_model)
        expected_slice = torch.tensor(
            [[0.0616, -0.5146, -0.4032], [-0.7629, -0.4934, -1.7153], [-0.4768, -0.6403, -0.7826]]
        )
        assert torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4)

    elif task == "object_detection":
        # coco has 91 labels
        config.num_labels = 91
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        # load model from torch hub
        if backbone == "resnet_50" and not dilation:
            detr = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True).eval()
        elif backbone == "resnet_50" and dilation:
            detr = torch.hub.load("facebookresearch/detr", "detr_dc5_resnet50", pretrained=True).eval()
            config.dilation = True
        elif backbone == "resnet_101" and not dilation:
            detr = torch.hub.load("facebookresearch/detr", "detr_resnet101", pretrained=True).eval()
            config.backbone = "resnet_101"
        elif backbone == "resnet_101" and dilation:
            detr = torch.hub.load("facebookresearch/detr", "detr_dc5_resnet101", pretrained=True).eval()
            config.backbone = "resnet_101"
            config.dilation = True
        else:
            raise ValueError(f"Not supported: {backbone} with {dilation}")

        state_dict = detr.state_dict()
        # rename keys
        for src, dest in rename_keys:
            rename_key(state_dict, src, dest)
        # query, key and value matrices need special treatment
        read_in_q_k_v(state_dict)
        # rename classification heads
        for src, dest in rename_keys_object_detection_model:
            rename_key(state_dict, src, dest)
        # important: we need to prepend "model." to each of the base model keys as DetrForObjectDetection calls the base model like this
        for key in state_dict.copy().keys():
            if not key.startswith("class_labels_classifier") and not key.startswith("bbox_predictor"):
                val = state_dict.pop(key)
                state_dict["model." + key] = val
        # finally, create model and load state dict
        model = DetrForObjectDetection(config).eval()
        model.load_state_dict(state_dict)
        # verify our conversion
        original_outputs = detr(img)
        outputs = model(img)
        assert torch.allclose(outputs.pred_logits, original_outputs["pred_logits"], atol=1e-4)
        assert torch.allclose(outputs.pred_boxes, original_outputs["pred_boxes"], atol=1e-4)

    elif task == "panoptic_segmentation":
        # First, load in original detr from torch hub
        if backbone == "resnet_50" and not dilation:
            detr, postprocessor = torch.hub.load(
                "facebookresearch/detr",
                "detr_resnet50_panoptic",
                pretrained=True,
                return_postprocessor=True,
                num_classes=250,
            )
            detr.eval()
        elif backbone == "resnet_50" and dilation:
            detr, postprocessor = torch.hub.load(
                "facebookresearch/detr",
                "detr_dc5_resnet50_panoptic",
                pretrained=True,
                return_postprocessor=True,
                num_classes=250,
            )
            detr.eval()
            config.dilation = True
        elif backbone == "resnet_101" and not dilation:
            detr, postprocessor = torch.hub.load(
                "facebookresearch/detr",
                "detr_resnet101_panoptic",
                pretrained=True,
                return_postprocessor=True,
                num_classes=250,
            )
            detr.eval()
            config.backbone = "resnet_101"
        else:
            print("Not supported:", backbone, dilation)

    else:
        print("Task not in list of supported tasks:", task)

    # Save model
    logger.info(f"Saving PyTorch model to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="base_model",
        type=str,
        help="""
    Task for which to convert a checkpoint. One of 'base_model', 'object_detection' or 'panoptic_segmentation'.
    """,
    )
    parser.add_argument(
        "--backbone", default="resnet_50", type=str, help="Which backbone to use. One of 'resnet50', 'resnet101'."
    )
    parser.add_argument("--dilation", default=False, action="store_true", help="Whether to apply dilated convolution.")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_detr_checkpoint(args.task, args.backbone, args.dilation, args.pytorch_dump_folder_path)
