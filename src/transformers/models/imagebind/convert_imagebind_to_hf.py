# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import regex as re
import torch

from transformers import (
    CLIPTokenizer,
    ImageBindConfig,
    ImageBindFeatureExtractor,
    ImageBindImageProcessor,
    ImageBindModel,
    ImageBindProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Vision
    r"modality_preprocessors\.vision\.cls_token": "vision_model.embeddings.cls_token",
    r"modality_preprocessors\.vision\.rgbt_stem\.proj\.1\.weight": "vision_model.embeddings.patch_embedding.projection.weight",
    r"modality_preprocessors\.vision\.pos_embedding_helper\.pos_embed": "vision_model.embeddings.position_embeddings",
    r"modality_heads\.vision\.0\.weight": "vision_model.layernorm.weight",
    r"modality_heads\.vision\.0\.bias": "vision_model.layernorm.bias",
    r"modality_heads\.vision\.2\.weight": "vision_projection.weight",
    r"modality_trunks\.vision\.pre_transformer_layer\.0\.weight": "vision_model.pre_layernorm.weight",
    r"modality_trunks\.vision\.pre_transformer_layer\.0\.bias": "vision_model.pre_layernorm.bias",
    # Text
    r"modality_preprocessors\.text\.pos_embed": "text_model.embeddings.position_embedding.weight",
    r"modality_preprocessors\.text\.token_embedding\.weight": "text_model.embeddings.token_embedding.weight",
    r"modality_heads\.text\.proj\.0\.weight": "text_model.layernorm.weight",
    r"modality_heads\.text\.proj\.0\.bias": "text_model.layernorm.bias",
    r"modality_heads\.text\.proj\.1\.weight": "text_projection.weight",
    r"modality_postprocessors\.text\.1\.log_logit_scale": "text_postprocessor.log_logit_scale",
    # Audio
    r"modality_preprocessors\.audio\.cls_token": "audio_model.embeddings.cls_token",
    r"modality_preprocessors\.audio\.rgbt_stem\.proj\.weight": "audio_model.embeddings.patch_embedding.projection.weight",
    r"modality_preprocessors\.audio\.rgbt_stem\.norm_layer\.weight": "audio_model.embeddings.patch_embedding.layernorm.weight",
    r"modality_preprocessors\.audio\.rgbt_stem\.norm_layer\.bias": "audio_model.embeddings.patch_embedding.layernorm.bias",
    r"modality_preprocessors\.audio\.pos_embedding_helper\.pos_embed": "audio_model.embeddings.position_embeddings",
    r"modality_heads\.audio\.0\.weight": "audio_model.layernorm.weight",
    r"modality_heads\.audio\.0\.bias": "audio_model.layernorm.bias",
    r"modality_heads\.audio\.2\.weight": "audio_projection.weight",
}


def rename_encoder_layers(config, modality):
    rename_keys = {}
    # fmt: off
    # Patterns for the keys
    key_patterns = [
        (r"attn\.in_proj_weight", f"{modality}_model.encoder.layers.{{layer_idx}}.self_attn.qkv_proj.weight"),
        (r"attn\.in_proj_bias", f"{modality}_model.encoder.layers.{{layer_idx}}.self_attn.qkv_proj.bias"),
        (r"attn\.out_proj\.weight", f"{modality}_model.encoder.layers.{{layer_idx}}.self_attn.out_proj.weight"),
        (r"attn\.out_proj\.bias", f"{modality}_model.encoder.layers.{{layer_idx}}.self_attn.out_proj.bias"),
        (r"norm_1\.weight", f"{modality}_model.encoder.layers.{{layer_idx}}.layernorm_before.weight"),
        (r"norm_1\.bias", f"{modality}_model.encoder.layers.{{layer_idx}}.layernorm_before.bias"),
        (r"mlp\.fc1\.weight", f"{modality}_model.encoder.layers.{{layer_idx}}.mlp.fc1.weight"),
        (r"mlp\.fc1\.bias", f"{modality}_model.encoder.layers.{{layer_idx}}.mlp.fc1.bias"),
        (r"mlp\.fc2\.weight", f"{modality}_model.encoder.layers.{{layer_idx}}.mlp.fc2.weight"),
        (r"mlp\.fc2\.bias", f"{modality}_model.encoder.layers.{{layer_idx}}.mlp.fc2.bias"),
        (r"norm_2\.weight", f"{modality}_model.encoder.layers.{{layer_idx}}.layernorm_after.weight"),
        (r"norm_2\.bias", f"{modality}_model.encoder.layers.{{layer_idx}}.layernorm_after.bias"),
    ]

    for layer_idx in range(config.num_hidden_layers):
        for old_pattern, new_pattern in key_patterns:
            rename_keys[f"modality_trunks.{modality}.blocks.{layer_idx}.{old_pattern}"] = new_pattern.format(layer_idx=layer_idx)

        if config.add_kv_bias:
            rename_keys[f"modality_trunks.{modality}.blocks.{layer_idx}.attn.bias_k"] = f"{modality}_model.encoder.layers.{layer_idx}.self_attn.k_bias"
            rename_keys[f"modality_trunks.{modality}.blocks.{layer_idx}.attn.bias_v"] = f"{modality}_model.encoder.layers.{layer_idx}.self_attn.v_bias"

    # fmt: on

    return rename_keys


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    vision_config = config.vision_config
    text_config = config.text_config
    audio_config = config.audio_config

    rename_keys = {}

    # fmt: off

    rename_keys.update(ORIGINAL_TO_CONVERTED_KEY_MAPPING)

    rename_keys.update(
        rename_encoder_layers(vision_config, "vision")
    )

    rename_keys.update(
        rename_encoder_layers(text_config, "text")
    )

    rename_keys.update(
        rename_encoder_layers(audio_config, "audio")
    )
    # fmt: on

    return rename_keys


def rename_model_keys(dct, rename_keys):
    renamed_dict = {}

    for key, value in dct.items():
        new_key = key
        for pattern, new_pattern in rename_keys.items():
            new_key = re.sub(pattern, new_pattern, new_key)
        renamed_dict[new_key] = value

    return renamed_dict


def reshape_text_position_embeddings(state_dict):
    # Need to convert from (1, contexc_length, hidden_size) -> (context_length, hidden_size)
    position_embeddings = state_dict["text_model.embeddings.position_embedding.weight"]
    state_dict["text_model.embeddings.position_embedding.weight"] = position_embeddings.squeeze(0)

    return state_dict


@torch.no_grad()
def convert_imagebind_checkpoint(args):
    model_name = args.model_name
    pytorch_dump_folder_path = args.pytorch_dump_folder_path
    push_to_hub = args.push_to_hub
    hub_repo_path = args.hub_repo_path

    config = ImageBindConfig()

    # Load original checkpoint
    checkpoint_url = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")

    # Rename keys
    new_state_dict = original_state_dict.copy()
    rename_keys = create_rename_keys(config)

    new_state_dict = rename_model_keys(new_state_dict, rename_keys)

    reshape_text_position_embeddings(new_state_dict)

    # Load HF model
    model = ImageBindModel(config)

    model.eval()
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("")
    print("Unexpected keys:", unexpected_keys)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    image_processor = ImageBindImageProcessor()
    feature_extractor = ImageBindFeatureExtractor()
    processor = ImageBindProcessor(image_processor, tokenizer, feature_extractor)

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor for {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and processor for {model_name} to hub at {hub_repo_path}")
        model.push_to_hub(hub_repo_path)
        processor.push_to_hub(hub_repo_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model-name",
        default="imagebind-huge",
        type=str,
        choices=["imagebind-huge"],
        help="Name of the ImageBind model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch-dump-folder-path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    parser.add_argument(
        "--hub-repo-path", default=None, type=str, help="Path of the repository to push the model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_imagebind_checkpoint(args)
