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
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from transformers import DeformableDetrImageProcessor, RfDetrConfig, RfDetrForObjectDetection


# Mapping of model names to their checkpoint files
HOSTED_MODELS = {
    "rf-detr-base": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    # below is a less converged model that may be better for finetuning but worse for inference
    "rf-detr-base-2": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "rf-detr-large": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
    "rf-detr-nano": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    "rf-detr-small": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
    "rf-detr-medium": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
}

# Model configurations for different sizes
BACKBONE_CONFIGS = {
    "nano": {
        "attention_probs_dropout_prob": 0.0,
        "drop_path_rate": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-06,
        "layerscale_value": 1.0,
        "mlp_ratio": 4,
        "num_attention_heads": 6,
        "num_channels": 3,
        "num_hidden_layers": 12,
        "qkv_bias": True,
        "use_swiglu_ffn": False,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 16,
        "num_windows": 2,
        "image_size": 384,
    },
    "small": {
        "attention_probs_dropout_prob": 0.0,
        "drop_path_rate": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-06,
        "layerscale_value": 1.0,
        "mlp_ratio": 4,
        "num_attention_heads": 6,
        "num_channels": 3,
        "num_hidden_layers": 12,
        "qkv_bias": True,
        "use_swiglu_ffn": False,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 16,
        "num_windows": 2,
        "image_size": 512,
    },
    "base": {
        "attention_probs_dropout_prob": 0.0,
        "drop_path_rate": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-06,
        "layerscale_value": 1.0,
        "mlp_ratio": 4,
        "num_attention_heads": 6,
        "num_channels": 3,
        "num_hidden_layers": 12,
        "qkv_bias": True,
        "use_swiglu_ffn": False,
        "out_features": ["stage2", "stage5", "stage8", "stage11"],
        "hidden_size": 384,
        "patch_size": 14,
        "num_windows": 4,
        "image_size": 518,
    },
    "medium": {
        "attention_probs_dropout_prob": 0.0,
        "drop_path_rate": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-06,
        "layerscale_value": 1.0,
        "mlp_ratio": 4,
        "num_attention_heads": 6,
        "num_channels": 3,
        "num_hidden_layers": 12,
        "qkv_bias": True,
        "use_swiglu_ffn": False,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 16,
        "num_windows": 2,
        "image_size": 576,
    },
    "large": {
        "attention_probs_dropout_prob": 0.0,
        "drop_path_rate": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-06,
        "layerscale_value": 1.0,
        "mlp_ratio": 4,
        "num_attention_heads": 12,
        "num_channels": 3,
        "num_hidden_layers": 12,
        "qkv_bias": True,
        "use_swiglu_ffn": False,
        "out_features": ["stage2", "stage5", "stage8", "stage11"],
        "hidden_size": 768,
        "patch_size": 14,
        "num_windows": 4,
        "image_size": 518,
    },
}

MODEL_CONFIGS = {
    "nano": {
        "d_model": 256,
        "decoder_layers": 2,
        "decoder_self_attention_heads": 8,
        "decoder_cross_attention_heads": 16,
        "decoder_n_points": 2,
        "projector_scale_factors": [1.0],
        "num_queries": 300,
    },
    "small": {
        "d_model": 256,
        "decoder_layers": 3,
        "decoder_self_attention_heads": 8,
        "decoder_cross_attention_heads": 16,
        "decoder_n_points": 2,
        "projector_scale_factors": [1.0],
        "num_queries": 300,
    },
    "base": {
        "d_model": 256,
        "decoder_layers": 3,
        "decoder_self_attention_heads": 8,
        "decoder_cross_attention_heads": 16,
        "decoder_n_points": 2,
        "projector_scale_factors": [1.0],
        "num_queries": 300,
    },
    "medium": {
        "d_model": 256,
        "decoder_layers": 4,
        "decoder_self_attention_heads": 8,
        "decoder_cross_attention_heads": 16,
        "decoder_n_points": 2,
        "projector_scale_factors": [1.0],
        "num_queries": 300,
    },
    "large": {
        "d_model": 384,
        "num_queries": 300,
        "decoder_layers": 3,
        "decoder_self_attention_heads": 12,
        "decoder_cross_attention_heads": 24,
        "decoder_n_points": 4,
        "projector_scale_factors": [2.0, 0.5],
    },
}

IMAGE_PROCESSORS = {
    "nano": {
        "do_resize": True,
        "size": (384, 384),
    },
    "small": {
        "do_resize": True,
        "size": (512, 512),
    },
    "base": {
        "do_resize": True,
        "size": (560, 560),
    },
    "medium": {
        "do_resize": True,
        "size": (576, 576),
    },
    "large": {
        "do_resize": True,
        "size": (560, 560),
    },
}

# Key mapping for converting original checkpoint keys to HuggingFace format
# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # backbone projector scaling layers, sampling layers are dealt with seperately depending on the config
    r"backbone.0.encoder.encoder": r"backbone.model",
    r"backbone.0.projector.stages_sampling.(\d+).(\d+).(\d+).(weight|bias)":                                            r"backbone.projector.scale_layers.\1.sampling_layers.\2.layers.\3.\4",
    r"backbone.0.projector.stages_sampling.(\d+).(\d+).(\d+).conv.(weight|bias)":                                       r"backbone.projector.scale_layers.\1.sampling_layers.\2.layers.\3.conv.\4",
    r"backbone.0.projector.stages_sampling.(\d+).(\d+).(\d+).bn":                                                       r"backbone.projector.scale_layers.\1.sampling_layers.\2.layers.\3.norm",
    r"backbone.0.projector.stages.(\d+).0.cv1.conv.(weight|bias)":                                                      r"backbone.projector.scale_layers.\1.projector_layer.conv1.conv.\2",
    r"backbone.0.projector.stages.(\d+).0.cv1.bn.(weight|bias|running_mean|running_var|num_batches_tracked)":           r"backbone.projector.scale_layers.\1.projector_layer.conv1.norm.\2",
    r"backbone.0.projector.stages.(\d+).0.cv2.conv.(weight|bias)":                                                      r"backbone.projector.scale_layers.\1.projector_layer.conv2.conv.\2",
    r"backbone.0.projector.stages.(\d+).0.cv2.bn.(weight|bias|running_mean|running_var|num_batches_tracked)":           r"backbone.projector.scale_layers.\1.projector_layer.conv2.norm.\2",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv1.conv.(weight|bias)":                                              r"backbone.projector.scale_layers.\1.projector_layer.bottlenecks.\2.conv1.conv.\3",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv1.bn.(weight|bias|running_mean|running_var|num_batches_tracked)":   r"backbone.projector.scale_layers.\1.projector_layer.bottlenecks.\2.conv1.norm.\3",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv2.conv.(weight|bias)":                                              r"backbone.projector.scale_layers.\1.projector_layer.bottlenecks.\2.conv2.conv.\3",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv2.bn.(weight|bias|running_mean|running_var|num_batches_tracked)":   r"backbone.projector.scale_layers.\1.projector_layer.bottlenecks.\2.conv2.norm.\3",
    r"backbone.0.projector.stages.(\d+).1.(weight|bias)":                                                               r"backbone.projector.scale_layers.\1.layer_norm.\2",

    # transformer decoder
    r"transformer.decoder.layers.(\d+).self_attn.out_proj.(weight|bias)":               r"decoder.layers.\1.self_attn.o_proj.\2",
    r"transformer.decoder.layers.(\d+).norm1.(weight|bias)":                            r"decoder.layers.\1.self_attn_layer_norm.\2",
    r"transformer.decoder.layers.(\d+).cross_attn.sampling_offsets.(weight|bias)":      r"decoder.layers.\1.cross_attn.sampling_offsets.\2",
    r"transformer.decoder.layers.(\d+).cross_attn.attention_weights.(weight|bias)":     r"decoder.layers.\1.cross_attn.attention_weights.\2",
    r"transformer.decoder.layers.(\d+).cross_attn.value_proj.(weight|bias)":            r"decoder.layers.\1.cross_attn.value_proj.\2",
    r"transformer.decoder.layers.(\d+).cross_attn.output_proj.(weight|bias)":           r"decoder.layers.\1.cross_attn.output_proj.\2",
    r"transformer.decoder.layers.(\d+).norm2.(weight|bias)":                            r"decoder.layers.\1.cross_attn_layer_norm.\2",
    r"transformer.decoder.layers.(\d+).linear1.(weight|bias)":                          r"decoder.layers.\1.mlp.fc1.\2",
    r"transformer.decoder.layers.(\d+).linear2.(weight|bias)":                          r"decoder.layers.\1.mlp.fc2.\2",
    r"transformer.decoder.layers.(\d+).norm3.(weight|bias)":                            r"decoder.layers.\1.layer_norm.\2",
    r"transformer.decoder.norm.(weight|bias)":                                          r"decoder.layernorm.\1",
    r"transformer.decoder.ref_point_head.layers.(\d+).(weight|bias)":                   r"decoder.ref_point_head.layers.\1.\2",

    r"transformer.enc_output.(\d+).(weight|bias)":                      r"enc_output.\1.\2",
    r"transformer.enc_output_norm.(\d+).(weight|bias)":                 r"enc_output_norm.\1.\2",
    r"transformer.enc_out_class_embed.(\d+).(weight|bias)":             r"enc_out_class_embed.\1.\2",
    r"transformer.enc_out_bbox_embed.(\d+).layers.(\d+).(weight|bias)": r"enc_out_bbox_embed.\1.layers.\2.\3",

    r"refpoint_embed.weight": r"reference_point_embed.weight",
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
        in_proj_bias = torch.cat([
            state_dict.pop(f"backbone.0.encoder.blocks.{i}.attn.q_bias"),
            state_dict.pop(f"backbone.0.encoder.blocks.{i}.attn.v_bias"),
        ])

        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"backbone.model.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.model.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[hidden_size:2*hidden_size, :]
        state_dict[f"backbone.model.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.model.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[:hidden_size]
        state_dict[f"backbone.model.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-hidden_size:]
    return state_dict

def backbone_convert(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model.backbone.0.encoder.encoder"):
            # backbone.0.encoder.encoder.embeddings...
            # backbone.0.encoder.encoder.encoder.layer...
            new_key = key.replace("model.backbone.0.encoder.encoder", "model.backbone.model")
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

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
    image_processor_config = None
    sizes = ["nano", "small", "medium", "base", "large"]
    for size in sizes:
        if size in model_name:
            config = MODEL_CONFIGS[size]
            config["backbone_config"] = BACKBONE_CONFIGS[size]
            image_processor_config = IMAGE_PROCESSORS[size]

    # Default to base configuration
    if config is None:
        config = MODEL_CONFIGS["base"]
        config["backbone_config"] = BACKBONE_CONFIGS["base"]
        image_processor_config = IMAGE_PROCESSORS["base"]
    config["backbone_config"]["model_type"] = "rf_detr_dinov2"

    if "objects365" in model_name:
        config["num_labels"] = 366
    elif "coco" in model_name :
        config["num_labels"] = 91
    else :
        config["num_labels"] = 91

    return config, image_processor_config

def get_backbone_projector_sampling_key_mapping(config: RfDetrConfig):
    key_mapping = {}
    for i, scale in enumerate(config.projector_scale_factors):
        if scale == 2.0:
            key_mapping.update({
                fr"backbone.0.projector.stages_sampling.{i}.(\d+).(\d+).(weight|bias)": fr"backbone.projector.scale_layers.{i}.sampling_layers.\1.layers.\2.\3",
            })
        elif scale == 0.5:
            key_mapping.update({
                fr"backbone.0.projector.stages_sampling.{i}.(\d+).(\d+).conv.weight":                                                   fr"backbone.projector.scale_layers.{i}.sampling_layers.\1.layers.\2.conv.weight",
                fr"backbone.0.projector.stages_sampling.{i}.(\d+).(\d+).bn.(weight|bias|running_mean|running_var|num_batches_tracked)": fr"backbone.projector.scale_layers.{i}.sampling_layers.\1.layers.\2.norm.\3",
            })
    return key_mapping

# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im

def original_preprocess_image(image, size):
    transform = transforms.Compose([
            transforms.Resize(list(size.values())),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image = transform(image)
    return image

def test_models_outputs(model: RfDetrForObjectDetection, image_processor: DeformableDetrImageProcessor, model_name: str):
    expected_outputs = {
    "rf-detr-nano": {
        "logits": [-6.68004, -5.66107, -11.70373, -8.32324, -5.76176],
        "boxes" : [0.25828, 0.54991, 0.47220, 0.87432, 0.55099],
    },
    "rf-detr-small": {
        "logits": [-6.83893, -4.55097, -10.53040, -8.20657, -5.55314],
        "boxes" : [0.25782, 0.55037, 0.47922, 0.87102, 0.77074],
    },
    "rf-detr-base": {
        "logits": [-7.60410, -4.65943, -10.03144, -5.63881, -9.88291],
        "boxes" : [0.25465, 0.54864, 0.48583, 0.86991, 0.16926],
    },
    "rf-detr-base-2": {
        "logits": [-6.81648, -6.80946, -7.72004, -6.06710, -10.37419],
        "boxes" : [0.16911, 0.19784, 0.21076, 0.09273, 0.25263],
    },
    "rf-detr-medium": {
        "logits": [-6.58581, -8.07883, -12.52392, -7.78248, -10.47323],
        "boxes" : [0.16824, 0.19932, 0.21110, 0.09385, 0.77087],
    },
    "rf-detr-large": {
        "logits": [-7.60888, -4.36906, -4.98865, -8.06598, -5.52970],
        "boxes" : [0.25576, 0.55051, 0.47765, 0.87141, 0.76966],
    },
}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = prepare_img()
    # Fake annotation for testing
    annotations = {
        "image_id": 0,
        "annotations": [
            {"bbox": [250, 250, 350, 350], "category_id": 0, "iscrowd": 0, "area": 122500}
        ],
    }

    original_pixel_values = original_preprocess_image(image, image_processor.size).unsqueeze(0).to(device)
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

    print("Forward pass looks ok!")

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers['content-length'])
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

@torch.no_grad()
def convert_rf_detr_checkpoint(
    model_name: str,
    checkpoint_url: str,
    pytorch_dump_folder_path: str,
    push_to_hub: bool = False,
    organization: str = "stevenbucaille",
):
    """
    Convert a RF-DETR checkpoint to HuggingFace format.

    Args:
        model_name: Name of the model (e.g., "lwdetr_tiny_30e_objects365")
        checkpoint_path: Path to the checkpoint file
        pytorch_dump_folder_path: Path to save the converted model
        push_to_hub: Whether to push the model to the hub
        organization: Organization to push the model to
    """
    print(f"Converting {model_name} checkpoint...")
    print(checkpoint_url)
    # Create output directory
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)

    # Get model configuration
    config, image_processor_config = get_model_config(model_name)
    lw_detr_config = RfDetrConfig(**config)

    # Save configuration
    lw_detr_config.save_pretrained(pytorch_dump_folder_path)
    print("Configuration saved successfully...")

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_url}...")
    checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", weights_only=False, file_name=model_name)
    # Create model and load weights
    print("Creating model and loading weights...")
    model = RfDetrForObjectDetection(lw_detr_config)

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
        # state_dict = backbone_read_in_q_k_v(state_dict, lw_detr_config)
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
        converted_state_dict = backbone_convert(converted_state_dict)

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
    image_processor = DeformableDetrImageProcessor(**image_processor_config, use_fast=True)
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
        "--model_name", type=str, required=True, choices=list(HOSTED_MODELS.keys()), help="Name of the model to convert"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", type=str, required=True, help="Path to the output PyTorch model directory"
    )
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file (if not using hub download)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to the hub")
    parser.add_argument("--organization", type=str, default="stevenbucaille", help="Organization to push the model to")

    args = parser.parse_args()

    # Get checkpoint path
    checkpoint_path = args.checkpoint_path


    # Convert checkpoint
    convert_rf_detr_checkpoint(
        model_name=args.model_name,
        checkpoint_url=checkpoint_path,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        push_to_hub=args.push_to_hub,
        organization=args.organization,
    )


if __name__ == "__main__":
    main()
