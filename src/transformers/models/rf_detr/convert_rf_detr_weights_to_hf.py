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
import json

import torch
from huggingface_hub import hf_hub_download

from transformers import (
    DetrImageProcessor,
    RfDetrConfig,
    RfDetrForInstanceSegmentation,
    RfDetrForObjectDetection,
)


# Mapping of model names to their checkpoint files
HOSTED_MODELS = {
    "rf-detr-base": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    # base-2 is a less converged model that may be better for finetuning but worse for inference
    "rf-detr-base-2": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "rf-detr-nano": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    "rf-detr-small": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
    "rf-detr-medium": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
    "rf-detr-large": "https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth",
    "rf-detr-seg-preview": "https://storage.googleapis.com/rfdetr/rf-detr-seg-preview.pt",
    "rf-detr-seg-nano": "https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth",
    "rf-detr-seg-small": "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth",
    "rf-detr-seg-medium": "https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth",
    "rf-detr-seg-large": "https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth",
    "rf-detr-seg-xlarge": "https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth",
    "rf-detr-seg-xxlarge": "https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth",
}

# Model configurations for different sizes

BASE_BACKBONE_CONFIG = {
    "attention_probs_dropout_prob": 0.0,
    "drop_path_rate": 0.0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 384,
    "image_size": 384,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-06,
    "layerscale_value": 1.0,
    "mlp_ratio": 4,
    "num_channels": 3,
    "num_hidden_layers": 12,
    "num_attention_heads": 6,
    "num_windows": 2,
    "out_features": ["stage3", "stage6", "stage9", "stage12"],
    "patch_size": 16,
    "qkv_bias": True,
    "use_swiglu_ffn": False,
}

# Backbone specs: model_name -> {image_size, **overrides of BASE_BACKBONE_CONFIG}
MODEL_VARIANT_BACKBONE_CONFIG_KWARGS = {
    "rf-detr-nano": {},
    "rf-detr-small": {"image_size": 512},
    "rf-detr-base": {
        "image_size": 518,
        "patch_size": 14,
        "num_windows": 4,
        "out_features": ["stage2", "stage5", "stage8", "stage11"],
    },
    "rf-detr-medium": {"image_size": 576},
    "rf-detr-large": {"image_size": 704},
    "rf-detr-seg-preview": {"image_size": 432, "patch_size": 12},
    "rf-detr-seg-nano": {"image_size": 312, "patch_size": 12, "num_windows": 1},
    "rf-detr-seg-small": {"patch_size": 12},
    "rf-detr-seg-medium": {"image_size": 432, "patch_size": 12},
    "rf-detr-seg-large": {"image_size": 504, "patch_size": 12},
    "rf-detr-seg-xlarge": {"image_size": 624, "patch_size": 12},
    "rf-detr-seg-xxlarge": {"image_size": 768, "patch_size": 12},
}

BACKBONE_CONFIGS = {
    name: {**BASE_BACKBONE_CONFIG, **spec} for name, spec in MODEL_VARIANT_BACKBONE_CONFIG_KWARGS.items()
}

BASE_MODEL_CONFIG = {
    "d_model": 256,
    "num_queries": 300,
    "decoder_self_attention_heads": 8,
    "decoder_cross_attention_heads": 16,
    "decoder_n_points": 2,
    "projector_scale_factors": [1.0],
    "intermediate_size": 1024,
}

# Model specs: model_name -> overrides of BASE_MODEL_CONFIG (at least decoder_layers)
MODEL_VARIANT_CONFIG_KWARGS = {
    "rf-detr-nano": {"decoder_layers": 2},
    "rf-detr-small": {"decoder_layers": 3},
    "rf-detr-base": {"decoder_layers": 3},
    "rf-detr-medium": {"decoder_layers": 4},
    "rf-detr-large": {"decoder_layers": 4},
    "rf-detr-seg-preview": {"decoder_layers": 4, "num_queries": 200, "class_loss_coefficient": 5.0},
    "rf-detr-seg-nano": {"decoder_layers": 4, "num_queries": 100, "class_loss_coefficient": 5.0},
    "rf-detr-seg-small": {"decoder_layers": 4, "num_queries": 100, "class_loss_coefficient": 5.0},
    "rf-detr-seg-medium": {"decoder_layers": 5, "num_queries": 200, "class_loss_coefficient": 5.0},
    "rf-detr-seg-large": {"decoder_layers": 5, "class_loss_coefficient": 5.0},
    "rf-detr-seg-xlarge": {"decoder_layers": 6, "class_loss_coefficient": 5.0},
    "rf-detr-seg-xxlarge": {"decoder_layers": 6, "class_loss_coefficient": 5.0},
}

MODEL_CONFIGS = {name: {**BASE_MODEL_CONFIG, **spec} for name, spec in MODEL_VARIANT_CONFIG_KWARGS.items()}

IMAGE_PROCESSORS = {
    "rf-detr-nano": (384, 384),
    "rf-detr-small": (512, 512),
    "rf-detr-base": (560, 560),
    "rf-detr-base-2": (560, 560),
    "rf-detr-medium": (576, 576),
    "rf-detr-large": (704, 704),
    "rf-detr-seg-preview": (432, 432),
    "rf-detr-seg-nano": (312, 312),
    "rf-detr-seg-small": (384, 384),
    "rf-detr-seg-medium": (432, 432),
    "rf-detr-seg-large": (504, 504),
    "rf-detr-seg-xlarge": (624, 624),
    "rf-detr-seg-xxlarge": (768, 768),
}


def get_model_config(model_name: str):
    """Get the appropriate configuration for a given model size."""
    config_key = model_name if model_name in MODEL_CONFIGS else "rf-detr-base"
    if config_key != model_name:
        print(f"No config found for {model_name}, using base config")
    else:
        print(f"Found config for {model_name}")

    config = MODEL_CONFIGS[config_key].copy()
    config["backbone_config"] = BACKBONE_CONFIGS[config_key].copy()
    config["backbone_config"]["model_type"] = "rf_detr_dinov2"

    if "objects365" in model_name:
        config["num_labels"] = 366
    else:
        config["num_labels"] = 91
        repo_id = "huggingface/label-files"
        filename = "coco-detection-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config["id2label"] = id2label
        config["label2id"] = {v: k for k, v in id2label.items()}

    return config


@torch.no_grad()
def convert_rf_detr_checkpoint(
    model_name: str,
    checkpoint_url: str,
    pytorch_dump_folder_path: str,
    push_to_hub: bool = False,
    organization: str = "Roboflow",
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
    config = get_model_config(model_name)
    rf_detr_config = RfDetrConfig(**config)

    # Load checkpoint
    checkpoint_url = checkpoint_url if checkpoint_url is not None else HOSTED_MODELS[model_name]
    print(f"Loading checkpoint from {checkpoint_url}...")
    checkpoint = torch.hub.load_state_dict_from_url(
        checkpoint_url, map_location="cpu", weights_only=False, file_name=f"{model_name}.pth"
    )
    # Create model and load weights
    print("Creating model and loading weights...")
    is_segmentation = "seg" in model_name
    model_class = RfDetrForInstanceSegmentation if is_segmentation else RfDetrForObjectDetection

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model, loading_info = model_class.from_pretrained(
        None, config=rf_detr_config, state_dict=state_dict, output_loading_info=True
    )
    print("Checkpoint loaded...")
    if (
        len(loading_info["missing_keys"]) > 0
        or len(loading_info["unexpected_keys"]) > 0
        or len(loading_info["mismatched_keys"]) > 0
    ):
        print("MISSING:", len(loading_info["missing_keys"]))
        print("\n".join(sorted(loading_info["missing_keys"])))
        print("UNEXPECTED:", len(loading_info["unexpected_keys"]))
        print("\n".join(sorted(loading_info["unexpected_keys"])))
        print("MISMATCH:", len(loading_info["mismatched_keys"]))
        print(loading_info["mismatched_keys"])

    image_processor = DetrImageProcessor(size=IMAGE_PROCESSORS[model_name], do_resize=True, use_fast=True)

    repo_id = f"{organization}/{model_name}"
    # Save model
    print("Saving model..." + " and pushing to hub..." if push_to_hub else "")
    model.save_pretrained(
        pytorch_dump_folder_path, push_to_hub=push_to_hub, repo_id=repo_id, commit_message=f"Add {model_name} model"
    )

    # Save image processor
    print("Saving image processor..." + " and pushing to hub..." if push_to_hub else "")
    image_processor.save_pretrained(
        pytorch_dump_folder_path,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        commit_message=f"Add {model_name} image processor",
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
