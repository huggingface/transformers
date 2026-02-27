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

import torch

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
    convert_and_load_state_dict_in_model,
)
from transformers.modeling_utils import LoadStateDictConfig


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
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-06,
    "layerscale_value": 1.0,
    "mlp_ratio": 4,
    "num_channels": 3,
    "num_hidden_layers": 12,
    "qkv_bias": True,
    "use_swiglu_ffn": False,
}

BACKBONE_CONFIGS = {
    "rf-detr-nano": {
        **BASE_BACKBONE_CONFIG,
        "num_attention_heads": 6,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 16,
        "num_windows": 2,
        "image_size": 384,
    },
    "rf-detr-small": {
        **BASE_BACKBONE_CONFIG,
        "num_attention_heads": 6,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 16,
        "num_windows": 2,
        "image_size": 512,
    },
    "rf-detr-base": {
        **BASE_BACKBONE_CONFIG,
        "num_attention_heads": 6,
        "out_features": ["stage2", "stage5", "stage8", "stage11"],
        "hidden_size": 384,
        "patch_size": 14,
        "num_windows": 4,
        "image_size": 518,
    },
    "rf-detr-medium": {
        **BASE_BACKBONE_CONFIG,
        "num_attention_heads": 6,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 16,
        "num_windows": 2,
        "image_size": 576,
    },
    "rf-detr-large": {
        **BASE_BACKBONE_CONFIG,
        "num_attention_heads": 6,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 16,
        "num_windows": 2,
        "image_size": 704,
    },
    "rf-detr-seg-preview": {
        **BASE_BACKBONE_CONFIG,
        "num_attention_heads": 6,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 12,
        "num_windows": 2,
        "image_size": 432,
    },
    "rf-detr-seg-nano": {
        **BASE_BACKBONE_CONFIG,
        "num_attention_heads": 6,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 12,
        "num_windows": 1,
        "image_size": 312,
    },
    "rf-detr-seg-small": {
        **BASE_BACKBONE_CONFIG,
        "num_attention_heads": 6,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 12,
        "num_windows": 2,
        "image_size": 384,
    },
    "rf-detr-seg-medium": {
        **BASE_BACKBONE_CONFIG,
        "num_attention_heads": 6,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 12,
        "num_windows": 2,
        "image_size": 432,
    },
    "rf-detr-seg-large": {
        **BASE_BACKBONE_CONFIG,
        "num_attention_heads": 6,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 12,
        "num_windows": 2,
        "image_size": 504,
    },
    "rf-detr-seg-xlarge": {
        **BASE_BACKBONE_CONFIG,
        "num_attention_heads": 6,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 12,
        "num_windows": 2,
        "image_size": 624,
    },
    "rf-detr-seg-xxlarge": {
        **BASE_BACKBONE_CONFIG,
        "num_attention_heads": 6,
        "out_features": ["stage3", "stage6", "stage9", "stage12"],
        "hidden_size": 384,
        "patch_size": 12,
        "num_windows": 2,
        "image_size": 768,
    },
}

BASE_MODEL_CONFIG = {
    "d_model": 256,
    "num_queries": 300,
    "decoder_self_attention_heads": 8,
    "decoder_cross_attention_heads": 16,
    "decoder_n_points": 2,
    "projector_scale_factors": [1.0],
}

MODEL_CONFIGS = {
    "rf-detr-nano": {
        **BASE_MODEL_CONFIG,
        "decoder_layers": 2,
    },
    "rf-detr-small": {
        **BASE_MODEL_CONFIG,
        "decoder_layers": 3,
    },
    "rf-detr-base": {
        **BASE_MODEL_CONFIG,
        "decoder_layers": 3,
    },
    "rf-detr-medium": {
        **BASE_MODEL_CONFIG,
        "decoder_layers": 4,
    },
    "rf-detr-large": {
        **BASE_MODEL_CONFIG,
        "decoder_layers": 4,
    },
    "rf-detr-seg-preview": {
        **BASE_MODEL_CONFIG,
        "decoder_layers": 4,
        "num_queries": 200,
        "class_loss_coefficient": 5.0,
    },
    "rf-detr-seg-nano": {
        **BASE_MODEL_CONFIG,
        "decoder_layers": 4,
        "num_queries": 100,
        "class_loss_coefficient": 5.0,
    },
    "rf-detr-seg-small": {
        **BASE_MODEL_CONFIG,
        "decoder_layers": 4,
        "num_queries": 100,
        "class_loss_coefficient": 5.0,
    },
    "rf-detr-seg-medium": {
        **BASE_MODEL_CONFIG,
        "decoder_layers": 5,
        "num_queries": 200,
        "class_loss_coefficient": 5.0,
    },
    "rf-detr-seg-large": {
        **BASE_MODEL_CONFIG,
        "decoder_layers": 5,
        "class_loss_coefficient": 5.0,
    },
    "rf-detr-seg-xlarge": {
        **BASE_MODEL_CONFIG,
        "decoder_layers": 6,
        "class_loss_coefficient": 5.0,
    },
    "rf-detr-seg-xxlarge": {
        **BASE_MODEL_CONFIG,
        "decoder_layers": 6,
        "class_loss_coefficient": 5.0,
    },
}

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
    config = None
    sizes = MODEL_CONFIGS.keys()
    for size in sizes:
        if size == model_name:
            print(f"Found config for {model_name}")
            config = MODEL_CONFIGS[size]
            config["backbone_config"] = BACKBONE_CONFIGS[size]
            break

    # Default to base configuration
    if config is None:
        print(f"No config found for {model_name}, using base config")
        config = MODEL_CONFIGS["rf-detr-base"]
        config["backbone_config"] = BACKBONE_CONFIGS["rf-detr-base"]

    config["backbone_config"]["model_type"] = "rf_detr_dinov2"

    if "objects365" in model_name:
        config["num_labels"] = 366
    elif "coco" in model_name:
        config["num_labels"] = 91
    else:
        config["num_labels"] = 91

    return config


def get_weight_mapping(
    is_segmentation: bool,
) -> list[WeightConverter | WeightRenaming]:
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
            WeightRenaming(
                r"projector.stages_sampling.(\d+).(\d+).(\d+)",
                r"projector.scale_layers.\1.sampling_layers.\2.layers.\3",
            ),
            WeightRenaming(
                r"projector.stages_sampling.(\d+).(\d+).(\d+).conv.weight",
                r"projector.scale_layers.\1.sampling_layers.\2.layers.\3.conv.weight",
            ),
            WeightRenaming(
                r"projector.stages_sampling.(\d+).(\d+).(\d+).bn",
                r"projector.scale_layers.\1.sampling_layers.\2.layers.\3.norm",
            ),
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
    if is_segmentation:
        model = RfDetrForInstanceSegmentation(rf_detr_config)
    else:
        model = RfDetrForObjectDetection(rf_detr_config)

    weight_mapping = get_weight_mapping(is_segmentation)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    load_config = LoadStateDictConfig(weight_mapping=weight_mapping)
    missing, unexpected, mismatch, _, misc = convert_and_load_state_dict_in_model(
        model, state_dict, load_config, tp_plan=None
    )
    print("Checkpoint loaded...")
    if len(missing) > 0 or len(unexpected) > 0 or len(mismatch) > 0:
        print("MISSING:", len(missing))
        print("\n".join(sorted(missing)))
        print("UNEXPECTED:", len(unexpected))
        print("\n".join(sorted(unexpected)))
        print("MISMATCH:", len(mismatch))
        print(mismatch)

    image_processor = DetrImageProcessor(size=IMAGE_PROCESSORS[model_name], do_resize=True, use_fast=True)

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
