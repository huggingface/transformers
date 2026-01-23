# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Convert RF-DETR checkpoints to transformers format."""

import argparse

import requests
import torch
from PIL import Image
from torchvision import transforms

from transformers import (
    DetrImageProcessor,
    RfDetrConfig,
    RfDetrForInstanceSegmentation,
    RfDetrForObjectDetection,
)
from transformers.core_model_loading import (
    Chunk,
    WeightConverter,
    WeightRenaming,
    WeightTransform,
    convert_and_load_state_dict_in_model,
)


# Mapping of model names to their checkpoint files
HOSTED_MODELS = {
    "rf-detr-base": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    # base-2 is a less converged model that may be better for finetuning but worse for inference
    "rf-detr-base-2": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "rf-detr-large": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
    "rf-detr-nano": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    "rf-detr-small": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
    "rf-detr-medium": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
    "rf-detr-segmentation": "https://storage.googleapis.com/rfdetr/rf-detr-seg-preview.pt",
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
    "segmentation": {
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
        "patch_size": 12,
        "num_windows": 2,
        "image_size": 432,
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
    "segmentation": {
        "d_model": 256,
        "decoder_layers": 4,
        "decoder_self_attention_heads": 8,
        "decoder_cross_attention_heads": 16,
        "decoder_n_points": 2,
        "projector_scale_factors": [1.0],
        "num_queries": 200,
        "class_loss_coefficient": 5.0,
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
    "segmentation": {
        "do_resize": True,
        "size": (432, 432),
    },
}


def get_model_config(model_name: str):
    """Get the appropriate configuration for a given model size."""
    config = None
    image_processor_config = None
    sizes = ["nano", "small", "medium", "base", "large", "segmentation"]
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
    elif "coco" in model_name:
        config["num_labels"] = 91
    else:
        config["num_labels"] = 91

    return config, image_processor_config


def get_weight_mapping(
    rf_detr_config: RfDetrConfig,
    is_segmentation: bool,
) -> list[WeightTransform]:
    if is_segmentation:
        weight_mapping = [
            # backbone RfDetrConvEncoder
            WeightRenaming("backbone.0.encoder.encoder", "rf_detr.model.backbone.backbone"),
            WeightRenaming("backbone.0.projector", "rf_detr.model.backbone.projector"),
            # RfDetrDecoder
            WeightRenaming("transformer.decoder", "rf_detr.model.decoder"),
            # RfDetrForObjectDetection
            WeightRenaming(r"transformer.enc_out_bbox_embed", r"rf_detr.model.enc_out_bbox_embed"),
            WeightRenaming(r"transformer.enc_output.(\d+)", r"rf_detr.model.enc_output.\1"),
            WeightRenaming(r"transformer.enc_output_norm.(\d+)", r"rf_detr.model.enc_output_norm.\1"),
            WeightRenaming(r"transformer.enc_out_class_embed.(\d+)", r"rf_detr.model.enc_out_class_embed.\1"),
            WeightRenaming(r"refpoint_embed.weight", r"rf_detr.model.reference_point_embed.weight"),
        ]
    else:
        weight_mapping = [
            # backbone RfDetrConvEncoder
            WeightRenaming("backbone.0.encoder.encoder", "backbone.backbone"),
            WeightRenaming("backbone.0.projector", "backbone.projector"),
            # RfDetrDecoder
            WeightRenaming("transformer.decoder", "decoder"),
            # RfDetrForObjectDetection
            WeightRenaming("transformer.enc_out_bbox_embed", "enc_out_bbox_embed"),
            WeightRenaming(r"transformer.enc_output.(\d+)", r"enc_output.\1"),
            WeightRenaming(r"transformer.enc_output_norm.(\d+)", r"enc_output_norm.\1"),
            WeightRenaming(r"transformer.enc_out_class_embed.(\d+)", r"enc_out_class_embed.\1"),
            WeightRenaming(r"refpoint_embed.weight", r"reference_point_embed.weight"),
        ]

    weight_mapping.extend(
        [
            # RfDetrConvEncoder
            ## RfDetrMultiScaleProjector
            WeightRenaming(r"projector.stages_sampling.(\d+)", r"projector.scale_layers.\1.sampling_layers"),
            WeightRenaming(r"projector.stages.(\d+).0", r"projector.scale_layers.\1.projector_layer"),
            WeightRenaming(r"projector.stages.(\d+).1", r"projector.scale_layers.\1.layer_norm"),
            ## RfDetrSamplingLayer
            WeightRenaming(r"sampling_layers.(\d+)", r"sampling_layers.\1.layers"),
            WeightRenaming(r"layers.(\d+).bn", r"layers.\1.norm"),
            ### RfDetrC2FLayer
            WeightRenaming(r"projector_layer.cv1.conv", r"projector_layer.conv1.conv"),
            WeightRenaming(r"projector_layer.cv1.bn", r"projector_layer.conv1.norm"),
            WeightRenaming(r"projector_layer.cv2.conv", r"projector_layer.conv2.conv"),
            WeightRenaming(r"projector_layer.cv2.bn", r"projector_layer.conv2.norm"),
            WeightRenaming(r"projector_layer.m.(\d+)", r"projector_layer.bottlenecks.\1"),
            #### RfDetrRepVggBlock
            WeightRenaming(r"bottlenecks.(\d+).cv1.conv", r"bottlenecks.\1.conv1.conv"),
            WeightRenaming(r"bottlenecks.(\d+).cv1.bn", r"bottlenecks.\1.conv1.norm"),
            WeightRenaming(r"bottlenecks.(\d+).cv2.conv", r"bottlenecks.\1.conv2.conv"),
            WeightRenaming(r"bottlenecks.(\d+).cv2.bn", r"bottlenecks.\1.conv2.norm"),
            # RfDetrDecoder
            ## RfDetrDecoderLayer
            WeightRenaming(r"decoder.layers.(\d+).norm1", r"decoder.layers.\1.self_attn_layer_norm"),
            WeightRenaming(r"decoder.layers.(\d+).norm2", r"decoder.layers.\1.cross_attn_layer_norm"),
            WeightRenaming(r"decoder.layers.(\d+).linear1", r"decoder.layers.\1.mlp.fc1"),
            WeightRenaming(r"decoder.layers.(\d+).linear2", r"decoder.layers.\1.mlp.fc2"),
            WeightRenaming(r"decoder.layers.(\d+).norm3", r"decoder.layers.\1.layer_norm"),
            WeightRenaming("decoder.norm", r"decoder.layernorm"),
            ### RfDetrAttention
            WeightRenaming(r"self_attn.out_proj", r"self_attn.o_proj"),
            WeightConverter(
                r"self_attn.in_proj_bias",
                [r"self_attn.q_proj.bias", r"self_attn.k_proj.bias", r"self_attn.v_proj.bias"],
                operations=[Chunk(dim=0)],
            ),
            WeightConverter(
                r"self_attn.in_proj_weight",
                [r"self_attn.q_proj.weight", r"self_attn.k_proj.weight", r"self_attn.v_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
        ]
    )

    # Indices depend on the value of projector_scale_factors
    for i, scale in enumerate(rf_detr_config.projector_scale_factors):
        if scale == 2.0:
            weight_mapping.append(
                WeightRenaming(
                    rf"projector.stages_sampling.{i}.(\d+).(\d+)",
                    rf"projector.scale_layers.{i}.sampling_layers.\1.layers.\2",
                )
            )
        elif scale == 0.5:
            weight_mapping.append(
                WeightRenaming(
                    rf"projector.stages_sampling.{i}.(\d+).(\d+).conv.weight",
                    rf"projector.scale_layers.{i}.sampling_layers.\1.layers.\2.conv.weight",
                )
            )
            weight_mapping.append(
                WeightRenaming(
                    rf"projector.stages_sampling.{i}.(\d+).(\d+).bn",
                    rf"projector.scale_layers.{i}.sampling_layers.\1.layers.\2.norm",
                )
            )

    if is_segmentation:
        weight_mapping.extend(
            [
                # RfDetrForObjectDetection
                WeightRenaming(r"bbox_embed.layers", "rf_detr.bbox_embed.layers"),
                WeightRenaming(r"class_embed.(weight|bias)", r"rf_detr.class_embed.\1"),
                WeightRenaming(r"query_feat.(weight|bias)", r"rf_detr.model.query_feat.\1"),
                # Segmentation head
                WeightRenaming(r"segmentation_head.blocks", r"blocks"),
                WeightRenaming("segmentation_head.spatial_features_proj", "spatial_features_proj"),
                WeightRenaming("segmentation_head.query_features_block", "query_features_block"),
                WeightRenaming("segmentation_head.query_features_proj", "query_features_proj"),
                WeightRenaming("segmentation_head.bias", "bias"),
                ## RfDetrSegmentationMLPBlock
                WeightRenaming("query_features_block.layers.0", "query_features_block.in_linear"),
                WeightRenaming("query_features_block.layers.2", "query_features_block.out_linear"),
                ## list[RfDetrSegmentationBlock]
                WeightRenaming(r"blocks.(\d+)", r"blocks.\1"),
                WeightRenaming(r"blocks.(\d+).norm", r"blocks.\1.layernorm"),
            ]
        )
    return weight_mapping


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


def original_preprocess_image(image, size):
    transform = transforms.Compose(
        [
            transforms.Resize(list(size.values())),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image)
    return image


def test_models_outputs(model: RfDetrForObjectDetection, image_processor: DetrImageProcessor, model_name: str):
    expected_outputs = {
        "rf-detr-nano": {
            "logits": [-6.68004, -5.66107, -11.70373, -8.32324, -5.76176],
            "boxes": [0.25828, 0.54991, 0.47220, 0.87432, 0.55099],
            "loss": 14.893259,
        },
        "rf-detr-small": {
            "logits": [-6.83893, -4.55097, -10.53040, -8.20657, -5.55314],
            "boxes": [0.25782, 0.55037, 0.47922, 0.87102, 0.77074],
            "loss": 19.771887,
        },
        "rf-detr-base": {
            "logits": [-7.60410, -4.65943, -10.03144, -5.63881, -9.88291],
            "boxes": [0.25465, 0.54864, 0.48583, 0.86991, 0.16926],
            "loss": 21.967346,
        },
        "rf-detr-base-2": {
            "logits": [-6.81648, -6.80946, -7.72004, -6.06710, -10.37419],
            "boxes": [0.16911, 0.19784, 0.21076, 0.09273, 0.25263],
            "loss": 21.532478,
        },
        "rf-detr-medium": {
            "logits": [-6.58581, -8.07883, -12.52392, -7.78248, -10.47323],
            "boxes": [0.16824, 0.19932, 0.21110, 0.09385, 0.77087],
            "loss": 26.337656,
        },
        "rf-detr-large": {
            "logits": [-7.60888, -4.36906, -4.98865, -8.06598, -5.52970],
            "boxes": [0.25576, 0.55051, 0.47765, 0.87141, 0.76966],
            "loss": 22.116581,
        },
        "rf-detr-segmentation": {
            "logits": [-7.05877, -4.23362, -6.54288, -8.13384, -6.36838],
            "boxes": [0.25603, 0.55164, 0.48340, 0.87798, 0.73861],
            "pred_masks": [-16.72129, -16.17153, -17.06426, -17.29409, -17.78559],
            "loss": 76.156754,
        },
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = prepare_img()
    # Fake annotation for testing
    annotations = {
        "image_id": 0,
        "annotations": [
            {
                "bbox": [250, 250, 350, 350],
                "category_id": 0,
                "iscrowd": 0,
                "area": 122500,
                "segments": [[0, 0, 0, 100, 100, 100, 100, 0]],
            }
        ],
    }
    is_segmentation = "segmentation" in model_name

    original_pixel_values = original_preprocess_image(image, image_processor.size).unsqueeze(0).to(device)
    inputs = image_processor(images=image, annotations=annotations, return_tensors="pt").to(device)
    inputs["labels"][0]["masks"] = torch.zeros((1, original_pixel_values.shape[-1], original_pixel_values.shape[-2]))
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

    if is_segmentation:
        predicted_mask_logits = outputs.pred_masks.flatten()[:5]
        expected_mask_logits = expected_outputs[model_name]["pred_masks"]
        torch.testing.assert_close(predicted_mask_logits, torch.Tensor(expected_mask_logits), rtol=5e-3, atol=5e-3)

    predicted_loss = outputs.loss
    expected_loss = expected_outputs[model_name]["loss"]
    torch.testing.assert_close(predicted_loss, torch.tensor(expected_loss), rtol=5e-3, atol=5e-3)

    print("Forward pass looks ok!")


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

    # Get model configuration
    config, image_processor_config = get_model_config(model_name)
    rf_detr_config = RfDetrConfig(**config)

    # Load checkpoint
    checkpoint_url = checkpoint_url if checkpoint_url is not None else HOSTED_MODELS[model_name]
    print(f"Loading checkpoint from {checkpoint_url}...")
    checkpoint = torch.hub.load_state_dict_from_url(
        checkpoint_url, map_location="cpu", weights_only=False, file_name=model_name
    )
    # Create model and load weights
    print("Creating model and loading weights...")
    is_segmentation = "segmentation" in model_name
    if is_segmentation:
        model = RfDetrForInstanceSegmentation(rf_detr_config)
    else:
        model = RfDetrForObjectDetection(rf_detr_config)

    weight_mapping = get_weight_mapping(rf_detr_config, is_segmentation)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    missing, unexpected, mismatch, _, misc = convert_and_load_state_dict_in_model(
        model, state_dict, weight_mapping, tp_plan=None, hf_quantizer=None
    )
    print("Checkpoint loaded...")
    if len(mismatch) > 0 or len(unexpected) > 0 or len(mismatch) > 0:
        print("MISSING:", len(missing))
        print("\n".join(sorted(missing)))
        print("UNEXPECTED:", len(unexpected))
        print("\n".join(sorted(unexpected)))
        print("MISMATCH:", len(mismatch))
        print(mismatch)

    image_processor = DetrImageProcessor(**image_processor_config, use_fast=True)
    test_models_outputs(model, image_processor, model_name)

    repo_id = f"{organization}/{model_name}"
    # Save model
    print("Saving model..." + " and pushing to hub..." if push_to_hub else "")
    model.save_pretrained(
        pytorch_dump_folder_path,
        save_original_format=False,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        commit_message=f"Add {model_name} model",
    )

    # Save image processor
    print("Saving image processor..." + " and pushing to hub..." if push_to_hub else "")
    image_processor.save_pretrained(
        pytorch_dump_folder_path,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        commit_message=f"Add {model_name} image processor",
    )

    # Save config
    print("Saving config..." + " and pushing to hub..." if push_to_hub else "")
    rf_detr_config.save_pretrained(
        pytorch_dump_folder_path, push_to_hub=push_to_hub, repo_id=repo_id, commit_message=f"Add {model_name} config"
    )

    if push_to_hub:
        print("Pushed model to hub successfully!")

    print(f"Conversion completed successfully for {model_name}!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(HOSTED_MODELS.keys()),
        help="Name of the model to convert",
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
