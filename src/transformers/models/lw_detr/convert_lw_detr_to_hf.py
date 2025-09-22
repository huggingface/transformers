# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Convert LW-DETR checkpoints from the original repository.

URL: https://huggingface.co/xbsu/LW-DETR/tree/main/pretrain_weights
"""

import argparse
import os
import re
from typing import Optional

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import LwDetrConfig, LwDetrForObjectDetection, LwDetrImageProcessor
from transformers.image_utils import load_image


# All available LW-DETR checkpoints from the Hugging Face repository
HUB_MODEL_REPO = "xbsu/LW-DETR"

# Mapping of model names to their checkpoint files
HUB_CHECKPOINTS = {
    # LW-DETR models trained on Objects365
    "lwdetr_tiny_30e_objects365": "LWDETR_tiny_30e_objects365.pth",
    "lwdetr_small_30e_objects365": "LWDETR_small_30e_objects365.pth",
    "lwdetr_medium_30e_objects365": "LWDETR_medium_30e_objects365.pth",
    "lwdetr_large_30e_objects365": "LWDETR_large_30e_objects365.pth",
    "lwdetr_xlarge_30e_objects365": "LWDETR_xlarge_30e_objects365.pth",
    # LW-DETR models trained on COCO
    "lwdetr_tiny_60e_coco": "LWDETR_tiny_60e_coco.pth",
    "lwdetr_small_60e_coco": "LWDETR_small_60e_coco.pth",
    "lwdetr_medium_60e_coco": "LWDETR_medium_60e_coco.pth",
    "lwdetr_large_60e_coco": "LWDETR_large_60e_coco.pth",
    "lwdetr_xlarge_60e_coco": "LWDETR_xlarge_60e_coco.pth",
}

# Model configurations for different sizes
BACKBONE_CONFIGS = {
    "tiny": {
        "image_size": 1024,
        "hidden_size": 192,
        "num_hidden_layers": 6,
        "num_attention_heads": 12,
        "window_block_indices": [0, 2, 4],
        "out_indices": [1, 3, 5],
    },
    "small": {
        "image_size": 1024,
        "hidden_size": 192,
        "num_hidden_layers": 10,
        "num_attention_heads": 12,
        "window_block_indices": [0, 1, 3, 6, 7, 9],
        "out_indices": [2, 4, 5, 9],
    },
    "medium": {
        "image_size": 1024,
        "hidden_size": 384,
        "num_hidden_layers": 10,
        "num_attention_heads": 12,
        "window_block_indices": [0, 1, 3, 6, 7, 9],
        "out_indices": [2, 4, 5, 9],
    },
    "large": {
        "image_size": 1024,
        "hidden_size": 384,
        "num_hidden_layers": 10,
        "num_attention_heads": 12,
        "window_block_indices": [0, 1, 3, 6, 7, 9],
        "out_indices": [2, 4, 5, 9],
    },
    "xlarge": {
        "image_size": 1024,
        "hidden_size": 768,
        "num_hidden_layers": 10,
        "num_attention_heads": 12,
        "window_block_indices": [0, 1, 3, 6, 7, 9],
        "out_indices": [2, 4, 5, 9],
    },
}

MODEL_CONFIGS = {
    "tiny": {
        "projector_scale_factors": [1.0],
        "num_queries": 100,
        "decoder_layers": 3,
        "decoder_self_attention_heads": 8,
        "decoder_cross_attention_heads": 16,
        "decoder_n_points": 2,
        "d_model": 256,
    },
    "small": {
        "projector_scale_factors": [1.0],
        "num_queries": 300,
        "decoder_layers": 3,
        "decoder_self_attention_heads": 8,
        "decoder_cross_attention_heads": 16,
        "decoder_n_points": 2,
        "d_model": 256,
    },
    "medium": {
        "projector_scale_factors": [1.0],
        "num_queries": 300,
        "decoder_layers": 3,
        "decoder_self_attention_heads": 8,
        "decoder_cross_attention_heads": 16,
        "decoder_n_points": 2,
        "d_model": 256,
    },
    "large": {
        "projector_scale_factors": [2.0, 0.5],
        "num_queries": 300,
        "decoder_layers": 3,
        "decoder_self_attention_heads": 12,
        "decoder_cross_attention_heads": 24,
        "decoder_n_points": 4,
        "d_model": 384,
    },
    "xlarge": {
        "projector_scale_factors": [2.0, 0.5],
        "num_queries": 300,
        "decoder_layers": 3,
        "decoder_self_attention_heads": 12,
        "decoder_cross_attention_heads": 24,
        "decoder_n_points": 4,
        "d_model": 384,
    },
}

# Key mapping for converting original checkpoint keys to HuggingFace format
# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # backbone encoder
    r"backbone.0.encoder.pos_embed":                            r"backbone.conv_encoder.model.embeddings.position_embeddings",
    r"backbone.0.encoder.patch_embed.proj":                     r"backbone.conv_encoder.model.embeddings.projection",
    r"backbone.0.encoder.blocks.(\d+).gamma_1":                 r"backbone.conv_encoder.model.encoder.layer.\1.gamma_1",
    r"backbone.0.encoder.blocks.(\d+).gamma_2":                 r"backbone.conv_encoder.model.encoder.layer.\1.gamma_2",
    r"backbone.0.encoder.blocks.(\d+).norm1.(weight|bias)":     r"backbone.conv_encoder.model.encoder.layer.\1.layernorm_before.\2",
    r"backbone.0.encoder.blocks.(\d+).attn.proj.(weight|bias)": r"backbone.conv_encoder.model.encoder.layer.\1.attention.output.\2",
    r"backbone.0.encoder.blocks.(\d+).norm2.(weight|bias)":     r"backbone.conv_encoder.model.encoder.layer.\1.layernorm_after.\2",
    r"backbone.0.encoder.blocks.(\d+).mlp.fc1.(weight|bias)":   r"backbone.conv_encoder.model.encoder.layer.\1.intermediate.fc1.\2",
    r"backbone.0.encoder.blocks.(\d+).mlp.fc2.(weight|bias)":   r"backbone.conv_encoder.model.encoder.layer.\1.intermediate.fc2.\2",

    # backbone projector scaling layers, sampling layers are dealt with seperately depending on the config
    r"backbone.0.projector.stages.(\d+).0.cv1.conv.(weight|bias)":                                                      r"backbone.conv_encoder.projector.scale_layers.\1.projector_layer.conv1.conv.\2",
    r"backbone.0.projector.stages.(\d+).0.cv1.bn.(weight|bias|running_mean|running_var|num_batches_tracked)":           r"backbone.conv_encoder.projector.scale_layers.\1.projector_layer.conv1.norm.\2",
    r"backbone.0.projector.stages.(\d+).0.cv2.conv.(weight|bias)":                                                      r"backbone.conv_encoder.projector.scale_layers.\1.projector_layer.conv2.conv.\2",
    r"backbone.0.projector.stages.(\d+).0.cv2.bn.(weight|bias|running_mean|running_var|num_batches_tracked)":           r"backbone.conv_encoder.projector.scale_layers.\1.projector_layer.conv2.norm.\2",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv1.conv.(weight|bias)":                                              r"backbone.conv_encoder.projector.scale_layers.\1.projector_layer.bottlenecks.\2.conv1.conv.\3",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv1.bn.(weight|bias|running_mean|running_var|num_batches_tracked)":   r"backbone.conv_encoder.projector.scale_layers.\1.projector_layer.bottlenecks.\2.conv1.norm.\3",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv2.conv.(weight|bias)":                                              r"backbone.conv_encoder.projector.scale_layers.\1.projector_layer.bottlenecks.\2.conv2.conv.\3",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv2.bn.(weight|bias|running_mean|running_var|num_batches_tracked)":   r"backbone.conv_encoder.projector.scale_layers.\1.projector_layer.bottlenecks.\2.conv2.norm.\3",
    r"backbone.0.projector.stages.(\d+).1.(weight|bias)":                                                               r"backbone.conv_encoder.projector.scale_layers.\1.layer_norm.\2",

    # transformer decoder
    r"transformer.decoder.layers.(\d+).self_attn.out_proj.(weight|bias)":               r"decoder.layers.\1.self_attn.o_proj.\2",
    r"transformer.decoder.layers.(\d+).norm1.(weight|bias)":                            r"decoder.layers.\1.self_attn_layer_norm.\2",
    r"transformer.decoder.layers.(\d+).cross_attn.sampling_offsets.(weight|bias)":      r"decoder.layers.\1.cross_attn.sampling_offsets.\2",
    r"transformer.decoder.layers.(\d+).cross_attn.attention_weights.(weight|bias)":     r"decoder.layers.\1.cross_attn.attention_weights.\2",
    r"transformer.decoder.layers.(\d+).cross_attn.value_proj.(weight|bias)":            r"decoder.layers.\1.cross_attn.value_proj.\2",
    r"transformer.decoder.layers.(\d+).cross_attn.output_proj.(weight|bias)":           r"decoder.layers.\1.cross_attn.output_proj.\2",
    r"transformer.decoder.layers.(\d+).norm2.(weight|bias)":                            r"decoder.layers.\1.cross_attn_layer_norm.\2",
    r"transformer.decoder.layers.(\d+).linear1.(weight|bias)":                          r"decoder.layers.\1.fc1.\2",
    r"transformer.decoder.layers.(\d+).linear2.(weight|bias)":                          r"decoder.layers.\1.fc2.\2",
    r"transformer.decoder.layers.(\d+).norm3.(weight|bias)":                            r"decoder.layers.\1.final_layer_norm.\2",
    r"transformer.decoder.norm.(weight|bias)":                                          r"decoder.layernorm.\1",
    r"transformer.decoder.ref_point_head.layers.(\d+).(weight|bias)":                   r"decoder.ref_point_head.layers.\1.\2",

    r"transformer.enc_output.(\d+).(weight|bias)":                      r"enc_output.\1.\2",
    r"transformer.enc_output_norm.(\d+).(weight|bias)":                 r"enc_output_norm.\1.\2",
    r"transformer.enc_out_class_embed.(\d+).(weight|bias)":             r"enc_out_class_embed.\1.\2",
    r"transformer.enc_out_bbox_embed.(\d+).layers.(\d+).(weight|bias)": r"enc_out_bbox_embed.\1.layers.\2.\3",
}


def convert_old_keys_to_new_keys(state_dict_keys: Optional[dict] = None, key_mapping: Optional[dict] = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in key_mapping.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict

def backbone_read_in_q_k_v(state_dict, config):
    hidden_size = config.backbone_config.hidden_size
    # backbone encoder self-attention layers
    for i in range(config.backbone_config.num_hidden_layers):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"backbone.0.encoder.blocks.{i}.attn.qkv.weight")
        if config.backbone_config.use_cae:
            in_proj_bias = torch.cat([
                state_dict.pop(f"backbone.0.encoder.blocks.{i}.attn.q_bias"),
                state_dict.pop(f"backbone.0.encoder.blocks.{i}.attn.v_bias"),
            ])
        else:
            in_proj_bias = state_dict.pop(f"backbone.0.encoder.blocks.{i}.attn.qkv.bias")

        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"backbone.conv_encoder.model.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.conv_encoder.model.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[hidden_size:2*hidden_size, :]
        state_dict[f"backbone.conv_encoder.model.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        if config.backbone_config.use_cae:
            state_dict[f"backbone.conv_encoder.model.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[:hidden_size]
            state_dict[f"backbone.conv_encoder.model.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-hidden_size:]
        else:
            state_dict[f"backbone.conv_encoder.model.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[:hidden_size]
            state_dict[f"backbone.conv_encoder.model.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[hidden_size:2*hidden_size]
            state_dict[f"backbone.conv_encoder.model.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-hidden_size:]
    return state_dict

def read_in_q_k_v(state_dict, config):
    d_model = config.d_model
    # transformer decoder self-attention layers
    for i in range(config.decoder_layers):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:d_model, :]
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:d_model]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[d_model:2*d_model, :]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[d_model:2*d_model]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-d_model:, :]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-d_model:]
    return state_dict

def get_model_config(model_name: str):
    """Get the appropriate configuration for a given model size."""
    config = None
    sizes = ["tiny", "small", "medium", "large", "xlarge"]
    for size in sizes:
        if size in model_name:
            config = MODEL_CONFIGS[size]
            config["backbone_config"] = BACKBONE_CONFIGS[size]

    # Default to base configuration
    if config is None:
        config = MODEL_CONFIGS["base"]
        config["backbone_config"] = BACKBONE_CONFIGS["base"]
    config["backbone_config"]["model_type"] = "lw_detr_vit"

    if "objects365" in model_name:
        config["num_labels"] = 366
    elif "coco" in model_name :
        config["num_labels"] = 91
    else :
        config["num_labels"] = 20

    return config

def get_backbone_projector_sampling_key_mapping(config: LwDetrConfig):
    key_mapping = {}
    for i, scale in enumerate(config.projector_scale_factors):
        if scale == 2.0:
            if config.backbone_config.hidden_size > 512:
                key_mapping.update({
                    fr"backbone.0.projector.stages_sampling.{i}.(\d+).0.conv.weight": fr"backbone.conv_encoder.projector.scale_layers.{i}.sampling_layers.\1.layers.0.conv.weight",
                    fr"backbone.0.projector.stages_sampling.{i}.(\d+).0.bn.(weight|bias|running_mean|running_var|num_batches_tracked)": fr"backbone.conv_encoder.projector.scale_layers.{i}.sampling_layers.\1.layers.0.norm.\2",
                    fr"backbone.0.projector.stages_sampling.{i}.(\d+).1.(weight|bias)": fr"backbone.conv_encoder.projector.scale_layers.{i}.sampling_layers.\1.layers.1.\2",
                })
            else:
                key_mapping.update({
                    fr"backbone.0.projector.stages_sampling.{i}.(\d+).(\d+).(weight|bias)": fr"backbone.conv_encoder.projector.scale_layers.{i}.sampling_layers.\1.layers.\2.\3",
                })
        elif scale == 0.5:
            key_mapping.update({
                fr"backbone.0.projector.stages_sampling.{i}.(\d+).(\d+).conv.weight":                                                   fr"backbone.conv_encoder.projector.scale_layers.{i}.sampling_layers.\1.layers.\2.conv.weight",
                fr"backbone.0.projector.stages_sampling.{i}.(\d+).(\d+).bn.(weight|bias|running_mean|running_var|num_batches_tracked)": fr"backbone.conv_encoder.projector.scale_layers.{i}.sampling_layers.\1.layers.\2.norm.\3",
            })
    return key_mapping

# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im

def original_preprocess_image(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    transform = transforms.Compose([
            transforms.Resize([640, 640]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image = transform(image)
    return image

def test_models_outputs(model: LwDetrForObjectDetection, image_processor: LwDetrImageProcessor, model_name: str):
    expected_outputs = {
    "lwdetr_tiny_30e_objects365": {
        "logits": [-7.98357, -3.25769, -4.27604, -4.36642, -4.26377],
        "boxes" : [0.69171, 0.41640, 0.42500, 0.71044, 0.68572],
        "loss" : 22.356031
    },
    "lwdetr_small_30e_objects365": {
        "logits": [-8.23090, -4.51939, -4.60962, -5.84116, -4.66924],
        "boxes" : [0.38197, 0.19453, 0.08374, 0.13949, 0.54836],
        "loss" : 25.783434
    },
    "lwdetr_medium_30e_objects365": {
        "logits": [-8.51335, -5.30388, -6.10901, -6.24950, -6.17019],
        "boxes" : [0.42670, 0.19626, 0.05662, 0.15986, 0.63693],
        "loss" : 19.916319
    },
    "lwdetr_large_30e_objects365": {
        "logits": [-9.14638, -4.36668, -5.90443, -5.44150, -5.60903],
        "boxes" : [0.69699, 0.34825, 0.59264, 0.60608, 0.59220],
        "loss" : 16.564697
    },
    "lwdetr_xlarge_30e_objects365": {
        "logits": [-9.43218, -5.09647, -5.15782, -6.08266, -5.65616],
        "boxes" : [0.38041, 0.28770, 0.24946, 0.33373, 0.39383],
        "loss" : 19.576864
    },
    "lwdetr_tiny_60e_coco": {
        "logits": [-8.19712, -3.48709, -5.15203, -5.53926, -5.80503],
        "boxes" : [0.82672, 0.21823, 0.33963, 0.42020, 0.87035],
        "loss" : 24.127630
    },
    "lwdetr_small_60e_coco": {
        "logits": [-9.54076, -5.57151, -7.78802, -6.84619, -7.69232],
        "boxes" : [0.17627, 0.25203, 0.04201, 0.02213, 0.75684],
        "loss" : 24.499529
    },
    "lwdetr_medium_60e_coco": {
        "logits": [-8.67373, -6.01139, -6.28883, -6.86015, -6.68997],
        "boxes" : [0.66225, 0.45126, 0.22419, 0.59799, 0.29980],
        "loss" : 21.776751
    },
    "lwdetr_large_60e_coco": {
        "logits": [-8.65256, -4.49404, -6.67789, -6.37513, -6.19211],
        "boxes" : [0.71610, 0.44929, 0.37450, 0.60622, 0.72333],
        "loss" : 19.682697
    },
    "lwdetr_xlarge_60e_coco": {
        "logits": [-9.71923, -5.30107, -7.22207, -6.81112, -7.69698],
        "boxes" : [0.57546, 0.36955, 0.63460, 0.32719, 0.52905],
        "loss" : 18.377769
    }
}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(image_path)
    # Fake annotation for testing
    annotations = {
        "image_id": 0,
        "annotations": [
            {"bbox": [250, 250, 350, 350], "category_id": 0, "iscrowd": 0, "area": 122500}
        ],
    }

    original_pixel_values = original_preprocess_image(image_path).unsqueeze(0).to(device)
    inputs = image_processor(images=image, annotations=annotations, return_tensors="pt").to(device)

    torch.testing.assert_close(original_pixel_values, inputs["pixel_values"], atol=1e-6, rtol=1e-6)
    print("Preprocessing looks ok!")

    model.to(device)
    model.eval()
    model.config._attn_implementation = "eager"
    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    predicted_logits = outputs.logits.flatten()[:5]
    expected_logits = expected_outputs[model_name]["logits"]
    predicted_boxes = outputs.pred_boxes.flatten()[:5]
    expected_boxes = expected_outputs[model_name]["boxes"]
    torch.testing.assert_close(predicted_logits, torch.Tensor(expected_logits), rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(predicted_boxes, torch.Tensor(expected_boxes), rtol=5e-3, atol=5e-3)

    expected_loss = torch.tensor(expected_outputs[model_name]["loss"])
    predicted_loss = outputs.loss
    torch.testing.assert_close(predicted_loss, expected_loss, rtol=5e-3, atol=5e-3)
    print("Forward pass looks ok!")


@torch.no_grad()
def convert_lw_detr_checkpoint(
    model_name: str,
    checkpoint_path: str,
    pytorch_dump_folder_path: str,
    push_to_hub: bool = False,
    organization: str = "huggingface",
):
    """
    Convert a LW-DETR checkpoint to HuggingFace format.

    Args:
        model_name: Name of the model (e.g., "lwdetr_tiny_30e_objects365")
        checkpoint_path: Path to the checkpoint file
        pytorch_dump_folder_path: Path to save the converted model
        push_to_hub: Whether to push the model to the hub
        organization: Organization to push the model to
    """
    print(f"Converting {model_name} checkpoint...")

    # Create output directory
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)

    # Get model configuration
    config = get_model_config(model_name)
    lw_detr_config = LwDetrConfig(**config)

    # Save configuration
    lw_detr_config.save_pretrained(pytorch_dump_folder_path)
    print("Configuration saved successfully...")

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Create model and load weights
    print("Creating model and loading weights...")
    model = LwDetrForObjectDetection(lw_detr_config)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Convert keys if needed
    if ORIGINAL_TO_CONVERTED_KEY_MAPPING:
        backbone_projector_sampling_key_mapping = get_backbone_projector_sampling_key_mapping(lw_detr_config)
        state_dict = backbone_read_in_q_k_v(state_dict, lw_detr_config)
        state_dict = read_in_q_k_v(state_dict, lw_detr_config)
        key_mapping = ORIGINAL_TO_CONVERTED_KEY_MAPPING | backbone_projector_sampling_key_mapping
        all_keys = list(state_dict.keys())
        new_keys = convert_old_keys_to_new_keys(all_keys, key_mapping)
        prefix = "model."
        converted_state_dict = {}
        for key in all_keys:
            if not any(key.startswith(prefix) for prefix in ["class_embed", "bbox_embed"]):
                new_key = new_keys[key]
                converted_state_dict[prefix + new_key] = state_dict[key]
            else:
                converted_state_dict[key] = state_dict[key]

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    # Save model
    print("Saving model...")
    model.save_pretrained(pytorch_dump_folder_path)

    # Save image processor
    print("Saving image processor...")
    image_processor = LwDetrImageProcessor(size={"height": 640, "width": 640})
    image_processor.save_pretrained(pytorch_dump_folder_path)

    test_models_outputs(model, image_processor, model_name)

    if push_to_hub:
        print("Pushing model to hub...")
        model.push_to_hub(repo_id=f"{organization}/{model_name}", commit_message=f"Add {model_name} model")
        lw_detr_config.push_to_hub(repo_id=f"{organization}/{model_name}", commit_message=f"Add {model_name} config")
        image_processor.push_to_hub(
            repo_id=f"{organization}/{model_name}", commit_message=f"Add {model_name} image processor"
        )
        print("Pushed model to hub successfully!")

    print(f"Conversion completed successfully for {model_name}!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, choices=list(HUB_CHECKPOINTS.keys()), help="Name of the model to convert"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", type=str, required=True, help="Path to the output PyTorch model directory"
    )
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file (if not using hub download)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to the hub")
    parser.add_argument("--organization", type=str, default="stevenbucaille", help="Organization to push the model to")

    args = parser.parse_args()

    # Get checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        # Download from hub
        repo_id = HUB_MODEL_REPO
        filename = HUB_CHECKPOINTS[args.model_name]
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename, subfolder="pretrain_weights")

    # Convert checkpoint
    convert_lw_detr_checkpoint(
        model_name=args.model_name,
        checkpoint_path=checkpoint_path,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        push_to_hub=args.push_to_hub,
        organization=args.organization,
    )


if __name__ == "__main__":
    main()
