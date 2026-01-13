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

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import DeformableDetrImageProcessor, LwDetrConfig, LwDetrForObjectDetection
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
        "out_indices": [2, 4, 6],
    },
    "small": {
        "image_size": 1024,
        "hidden_size": 192,
        "num_hidden_layers": 10,
        "num_attention_heads": 12,
        "window_block_indices": [0, 1, 3, 6, 7, 9],
        "out_indices": [3, 5, 6, 10],
    },
    "medium": {
        "image_size": 1024,
        "hidden_size": 384,
        "num_hidden_layers": 10,
        "num_attention_heads": 12,
        "window_block_indices": [0, 1, 3, 6, 7, 9],
        "out_indices": [3, 5, 6, 10],
    },
    "large": {
        "image_size": 1024,
        "hidden_size": 384,
        "num_hidden_layers": 10,
        "num_attention_heads": 12,
        "window_block_indices": [0, 1, 3, 6, 7, 9],
        "out_indices": [3, 5, 6, 10],
    },
    "xlarge": {
        "image_size": 1024,
        "hidden_size": 768,
        "num_hidden_layers": 10,
        "num_attention_heads": 12,
        "window_block_indices": [0, 1, 3, 6, 7, 9],
        "out_indices": [3, 5, 6, 10],
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
    r"backbone.0.encoder.pos_embed":                            r"backbone.backbone.embeddings.position_embeddings",
    r"backbone.0.encoder.patch_embed.proj":                     r"backbone.backbone.embeddings.projection",
    r"backbone.0.encoder.blocks.(\d+).gamma_1":                 r"backbone.backbone.encoder.layer.\1.gamma_1",
    r"backbone.0.encoder.blocks.(\d+).gamma_2":                 r"backbone.backbone.encoder.layer.\1.gamma_2",
    r"backbone.0.encoder.blocks.(\d+).norm1.(weight|bias)":     r"backbone.backbone.encoder.layer.\1.layernorm_before.\2",
    r"backbone.0.encoder.blocks.(\d+).attn.proj.(weight|bias)": r"backbone.backbone.encoder.layer.\1.attention.output.\2",
    r"backbone.0.encoder.blocks.(\d+).norm2.(weight|bias)":     r"backbone.backbone.encoder.layer.\1.layernorm_after.\2",
    r"backbone.0.encoder.blocks.(\d+).mlp.fc1.(weight|bias)":   r"backbone.backbone.encoder.layer.\1.intermediate.fc1.\2",
    r"backbone.0.encoder.blocks.(\d+).mlp.fc2.(weight|bias)":   r"backbone.backbone.encoder.layer.\1.intermediate.fc2.\2",

    # backbone projector scaling layers, sampling layers are dealt with separately depending on the config
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

def delete_positional_embeddings_keys(state_dict):
    key_prefix = "backbone.0.encoder.pos_embed"
    keys_to_delete = [key for key in state_dict.keys() if key.startswith(key_prefix)]
    for key in keys_to_delete:
        del state_dict[key]
    return state_dict

def convert_old_keys_to_new_keys(state_dict_keys: dict | None = None, key_mapping: dict | None = None):
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
        state_dict[f"backbone.backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[hidden_size:2*hidden_size, :]
        state_dict[f"backbone.backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.backbone.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[:hidden_size]
        state_dict[f"backbone.backbone.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-hidden_size:]
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
                    fr"backbone.0.projector.stages_sampling.{i}.(\d+).0.conv.weight": fr"backbone.projector.scale_layers.{i}.sampling_layers.\1.layers.0.conv.weight",
                    fr"backbone.0.projector.stages_sampling.{i}.(\d+).0.bn.(weight|bias|running_mean|running_var|num_batches_tracked)": fr"backbone.projector.scale_layers.{i}.sampling_layers.\1.layers.0.norm.\2",
                    fr"backbone.0.projector.stages_sampling.{i}.(\d+).1.(weight|bias)": fr"backbone.projector.scale_layers.{i}.sampling_layers.\1.layers.1.\2",
                })
            else:
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

def original_preprocess_image(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    transform = transforms.Compose([
            transforms.Resize([640, 640]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image = transform(image)
    return image

def test_models_outputs(model: LwDetrForObjectDetection, image_processor: DeformableDetrImageProcessor, model_name: str):
    expected_outputs = {
    "lwdetr_tiny_30e_objects365": {
        "logits": [-7.76931, -4.12702, -2.90175, -4.06055, -2.95753],
        "boxes" : [0.16942, 0.19788, 0.21209, 0.09115, 0.25367],
        "loss" : 24.723873
    },
    "lwdetr_small_30e_objects365": {
        "logits": [-9.07081, -3.63061, -4.80831, -4.82828, -4.74330],
        "boxes" : [0.25428, 0.55128, 0.48572, 0.87628, 0.16972],
        "loss" : 21.224188
    },
    "lwdetr_medium_30e_objects365": {
        "logits": [-9.59531, -5.72241, -4.86421, -4.93863, -5.22668],
        "boxes" : [0.16904, 0.19795, 0.20996, 0.09260, 0.55067],
        "loss" : 21.400263
    },
    "lwdetr_large_30e_objects365": {
        "logits": [-9.34162, -4.38863, -3.78537, -5.50365, -4.84258],
        "boxes" : [0.25067, 0.55070, 0.48561, 0.87355, 0.77104],
        "loss" : 21.000555
    },
    "lwdetr_xlarge_30e_objects365": {
        "logits": [-11.92917, -4.33070, -4.40749, -5.02067, -6.92108],
        "boxes" : [0.76880, 0.41065, 0.46179, 0.72450, 0.25261],
        "loss" : 23.071398
    },
    "lwdetr_tiny_60e_coco": {
        "logits": [-9.07899, -4.66060, -7.37889, -5.83723, -7.13007],
        "boxes" : [0.76620, 0.41548, 0.46841, 0.72584, 0.25479],
        "loss" : 22.522924
    },
    "lwdetr_small_60e_coco": {
        "logits": [-9.53700, -5.22078, -7.97227, -7.72818, -6.39198],
        "boxes" : [0.76843, 0.41402, 0.46439, 0.72585, 0.25579],
        "loss" : 20.688507
    },
    "lwdetr_medium_60e_coco": {
        "logits": [-7.99542, -6.57918, -7.40989, -6.58970, -6.07415],
        "boxes" : [0.25352, 0.55045, 0.48459, 0.86760, 0.16862],
        "loss" : 19.571205
    },
    "lwdetr_large_60e_coco": {
        "logits": [-9.72590, -4.71623, -6.59528, -5.74079, -4.07018],
        "boxes" : [0.25423, 0.55063, 0.48317, 0.87475, 0.76970],
        "loss" : 19.438395
    },
    "lwdetr_xlarge_60e_coco": {
        "logits": [-11.17268, -4.82377, -7.55714, -7.85695, -7.94252],
        "boxes" : [0.25223, 0.54851, 0.47982, 0.87163, 0.77008],
        "loss" : 20.964485
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
    image_processor = DeformableDetrImageProcessor(size={"height": 640, "width": 640})
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
