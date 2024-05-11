# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import argparse

import torch

# from imagebind import load
from transformers import ImageBindConfig, ImageBindModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_encoder_layers(config, modality):
    rename_keys = []
    for layer_idx in range(config.num_hidden_layers):
        rename_keys.extend(
            [
                (
                    f"modality_trunks.{modality}.blocks.{layer_idx}.attn.in_proj_weight",
                    f"{modality}_model.encoder.layers.{layer_idx}.self_attn.qkv_proj.weight",
                ),
                (
                    f"modality_trunks.{modality}.blocks.{layer_idx}.attn.in_proj_bias",
                    f"{modality}_model.encoder.layers.{layer_idx}.self_attn.qkv_proj.bias",
                ),
                (
                    f"modality_trunks.{modality}.blocks.{layer_idx}.attn.out_proj.weight",
                    f"{modality}_model.encoder.layers.{layer_idx}.self_attn.out_proj.weight",
                ),
                (
                    f"modality_trunks.{modality}.blocks.{layer_idx}.attn.out_proj.bias",
                    f"{modality}_model.encoder.layers.{layer_idx}.self_attn.out_proj.bias",
                ),
                (
                    f"modality_trunks.{modality}.blocks.{layer_idx}.norm_1.weight",
                    f"{modality}_model.encoder.layers.{layer_idx}.layernorm_before.weight",
                ),
                (
                    f"modality_trunks.{modality}.blocks.{layer_idx}.norm_1.bias",
                    f"{modality}_model.encoder.layers.{layer_idx}.layernorm_before.bias",
                ),
                (
                    f"modality_trunks.{modality}.blocks.{layer_idx}.mlp.fc1.weight",
                    f"{modality}_model.encoder.layers.{layer_idx}.mlp.fc1.weight",
                ),
                (
                    f"modality_trunks.{modality}.blocks.{layer_idx}.mlp.fc1.bias",
                    f"{modality}_model.encoder.layers.{layer_idx}.mlp.fc1.bias",
                ),
                (
                    f"modality_trunks.{modality}.blocks.{layer_idx}.mlp.fc2.weight",
                    f"{modality}_model.encoder.layers.{layer_idx}.mlp.fc2.weight",
                ),
                (
                    f"modality_trunks.{modality}.blocks.{layer_idx}.mlp.fc2.bias",
                    f"{modality}_model.encoder.layers.{layer_idx}.mlp.fc2.bias",
                ),
                (
                    f"modality_trunks.{modality}.blocks.{layer_idx}.norm_2.weight",
                    f"{modality}_model.encoder.layers.{layer_idx}.layernorm_after.weight",
                ),
                (
                    f"modality_trunks.{modality}.blocks.{layer_idx}.norm_2.bias",
                    f"{modality}_model.encoder.layers.{layer_idx}.layernorm_after.bias",
                ),
            ]
        )
        if config.add_kv_bias:
            rename_keys.extend(
                [
                    (
                        f"modality_trunks.{modality}.blocks.{layer_idx}.attn.bias_k",
                        f"{modality}_model.encoder.layers.{layer_idx}.self_attn.k_bias",
                    ),
                    (
                        f"modality_trunks.{modality}.blocks.{layer_idx}.attn.bias_v",
                        f"{modality}_model.encoder.layers.{layer_idx}.self_attn.v_bias",
                    ),
                ]
            )

    return rename_keys


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    vision_config = config.vision_config
    text_config = config.text_config
    audio_config = config.audio_config

    rename_keys = []

    # fmt: off

    # Convert Vision
    rename_keys.extend([
        ("modality_preprocessors.vision.cls_token", "vision_model.embeddings.cls_token"),
        ("modality_preprocessors.vision.rgbt_stem.proj.1.weight", "vision_model.embeddings.patch_embedding.projection.weight"),
        ("modality_preprocessors.vision.pos_embedding_helper.pos_embed", "vision_model.embeddings.position_embedding"),
        ("modality_heads.vision.0.weight", "vision_model.layernorm.weight"),
        ("modality_heads.vision.0.bias", "vision_model.layernorm.bias"),
        ("modality_heads.vision.2.weight", "visual_projection.weight"),
        ("modality_trunks.vision.pre_transformer_layer.0.weight", "vision_model.pre_layernorm.weight"),
        ("modality_trunks.vision.pre_transformer_layer.0.bias", "vision_model.pre_layernorm.bias"),
    ])

    rename_keys.extend(
        rename_encoder_layers(vision_config, "vision")
    )

    # Convert Text
    rename_keys.extend([
        ("modality_preprocessors.text.pos_embed", "text_model.embeddings.position_embedding.weight"),
        ("modality_preprocessors.text.token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
        ("modality_heads.text.proj.0.weight", "text_model.layernorm.weight"),
        ("modality_heads.text.proj.0.bias", "text_model.layernorm.bias"),
        ("modality_heads.text.proj.1.weight", "text_projection.weight"),
        ("modality_postprocessors.text.1.log_logit_scale", "text_postprocessor.log_logit_scale"),
    ])

    rename_keys.extend(
        rename_encoder_layers(text_config, "text")
    )

    # Convert Audio
    rename_keys.extend([
        ("modality_preprocessors.audio.cls_token", "audio_model.embeddings.cls_token"),
        ("modality_preprocessors.audio.rgbt_stem.proj.weight", "audio_model.embeddings.patch_embedding.projection.weight"),
        ("modality_preprocessors.audio.rgbt_stem.norm_layer.weight", "audio_model.embeddings.patch_embedding.layernorm.weight"),
        ("modality_preprocessors.audio.rgbt_stem.norm_layer.bias", "audio_model.embeddings.patch_embedding.layernorm.bias"),
        ("modality_preprocessors.audio.pos_embedding_helper.pos_embed", "audio_model.embeddings.position_embedding"),
        ("modality_heads.audio.0.weight", "audio_model.layernorm.weight"),
        ("modality_heads.audio.0.bias", "audio_model.layernorm.bias"),
        ("modality_heads.audio.2.weight", "audio_projection.weight"),
    ])

    rename_keys.extend(
        rename_encoder_layers(audio_config, "audio")
    )
    # fmt: on

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def reshape_text_position_embedding(state_dict):
    # Need to convert from (1, contexc_length, hidden_size) -> (context_length, hidden_size)
    position_embedding = state_dict["text_model.embeddings.position_embedding.weight"]
    state_dict["text_model.embeddings.position_embedding.weight"] = position_embedding.squeeze(0)

    return state_dict


# We will verify our results on spongebob images
def prepare_input():
    ...


@torch.no_grad()
def convert_seggpt_checkpoint(args):
    model_name = args.model_name
    pytorch_dump_folder_path = args.pytorch_dump_folder_path
    verify_logits = args.verify_logits
    push_to_hub = args.push_to_hub

    config = ImageBindConfig()

    # Load original checkpoint
    checkpoint_url = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")

    # # Rename keys
    new_state_dict = original_state_dict.copy()
    rename_keys = create_rename_keys(config)

    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)
    reshape_text_position_embedding(new_state_dict)

    # Load HF model
    model = ImageBindModel(config)
    model.eval()
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("")
    print("Unexpected keys:", unexpected_keys)

    prepare_input()

    if verify_logits:
        expected_output = ...
        print(expected_output)
        print("Looks good!")
    else:
        print("Converted without verifying logits")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor for {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and processor for {model_name} to hub")
        model.push_to_hub(f"EduardoPacheco/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="imagebind-huge",
        type=str,
        choices=["imagebind-huge"],
        help="Name of the ImageBind model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--verify_logits",
        action="store_true",
        help="Whether or not to verify the logits against the original implementation.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_seggpt_checkpoint(args)
