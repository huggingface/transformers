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
"""Convert Table Transformer checkpoints with native (Transformers) backbone.

URL: https://github.com/microsoft/table-transformer
"""

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import functional as F

from transformers import DetrImageProcessor, ResNetConfig, TableTransformerConfig, TableTransformerForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


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
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.conv1.weight",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.convolution.weight",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn1.weight",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.normalization.weight",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn1.bias",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.normalization.bias",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn1.running_mean",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.normalization.running_mean",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn1.running_var",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.normalization.running_var",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.conv2.weight",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.convolution.weight",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn2.weight",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.normalization.weight",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn2.bias",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.normalization.bias",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn2.running_mean",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.normalization.running_mean",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn2.running_var",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.normalization.running_var",
                )
            )
            # all ResNet stages except the first one have a downsample as first layer
            if stage_idx != 0 and layer_idx == 0:
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
                        # "backbone.conv_encoder.model.encoder.stages.3.layers.0.shortcut.normalization.running_var"
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.1.running_var",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.running_var",
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
            ("transformer.encoder.norm.weight", "encoder.layernorm.weight"),
            ("transformer.encoder.norm.bias", "encoder.layernorm.bias"),
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


def resize(image, checkpoint_url):
    width, height = image.size
    current_max_size = max(width, height)
    target_max_size = 800 if "detection" in checkpoint_url else 1000
    scale = target_max_size / current_max_size
    resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

    return resized_image


def normalize(image):
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image


@torch.no_grad()
def convert_table_transformer_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DETR structure.
    """

    logger.info("Converting model...")

    # create HuggingFace model and load state dict
    backbone_config = ResNetConfig.from_pretrained(
        "microsoft/resnet-18", out_features=["stage1", "stage2", "stage3", "stage4"]
    )

    config = TableTransformerConfig(
        backbone_config=backbone_config,
        use_timm_backbone=False,
        mask_loss_coefficient=1,
        dice_loss_coefficient=1,
        ce_loss_coefficient=1,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.4,
        class_cost=1,
        bbox_cost=5,
        giou_cost=2,
    )

    # load original state dict
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")

    # rename keys
    for src, dest in create_rename_keys(config):
        rename_key(state_dict, src, dest)
    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    prefix = "model."
    for key in state_dict.copy():
        if not key.startswith("class_labels_classifier") and not key.startswith("bbox_predictor"):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val

    if "detection" in checkpoint_url:
        config.num_queries = 15
        config.num_labels = 2
        id2label = {0: "table", 1: "table rotated"}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    else:
        config.num_queries = 125
        config.num_labels = 6
        id2label = {
            0: "table",
            1: "table column",
            2: "table row",
            3: "table column header",
            4: "table projected row header",
            5: "table spanning cell",
        }
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    image_processor = DetrImageProcessor(format="coco_detection", size={"longest_edge": 800})
    model = TableTransformerForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    # verify our conversion
    filename = "example_pdf.png" if "detection" in checkpoint_url else "example_table.png"
    file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename=filename)
    image = Image.open(file_path).convert("RGB")
    pixel_values = normalize(resize(image, checkpoint_url)).unsqueeze(0)

    outputs = model(pixel_values)

    if "detection" in checkpoint_url:
        expected_shape = (1, 15, 3)
        expected_logits = torch.tensor(
            [[-6.7897, -16.9985, 6.7937], [-8.0186, -22.2192, 6.9677], [-7.3117, -21.0708, 7.4055]]
        )
        expected_boxes = torch.tensor([[0.4867, 0.1767, 0.6732], [0.6718, 0.4479, 0.3830], [0.4716, 0.1760, 0.6364]])

    else:
        expected_shape = (1, 125, 7)
        expected_logits = torch.tensor(
            [[-18.1430, -8.3214, 4.8274], [-18.4685, -7.1361, -4.2667], [-26.3693, -9.3429, -4.9962]]
        )
        expected_boxes = torch.tensor([[0.4983, 0.5595, 0.9440], [0.4916, 0.6315, 0.5954], [0.6108, 0.8637, 0.1135]])

    assert outputs.logits.shape == expected_shape
    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits, atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes, atol=1e-4)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        # Save model and image processor
        logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # Push model to HF hub
        logger.info("Pushing model to the hub...")
        model_name = (
            "microsoft/table-transformer-detection"
            if "detection" in checkpoint_url
            else "microsoft/table-transformer-structure-recognition"
        )
        model.push_to_hub(model_name, revision="no_timm")
        image_processor.push_to_hub(model_name, revision="no_timm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://pubtables1m.blob.core.windows.net/model/pubtables1m_detection_detr_r18.pth",
        type=str,
        choices=[
            "https://pubtables1m.blob.core.windows.net/model/pubtables1m_detection_detr_r18.pth",
            "https://pubtables1m.blob.core.windows.net/model/pubtables1m_structure_detr_r18.pth",
        ],
        help="URL of the Table Transformer checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()
    convert_table_transformer_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
