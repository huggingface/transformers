# coding=utf-8
# Copyright 2025 Microsoft and the HuggingFace Team. All rights reserved.
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
from collections import OrderedDict

import torch

from transformers import (
    AddedToken,
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    Florence2Config,
    Florence2ForConditionalGeneration,
    Florence2Processor,
    Florence2VisionConfig,
)


def convert_config(original_config: dict):
    new_config = Florence2VisionConfig(
        embed_dim=original_config["dim_embed"],
        max_temporal_embeddings=original_config["visual_temporal_embedding"]["max_temporal_embeddings"],
        max_pos_embeddings=original_config["image_pos_embed"]["max_pos_embeddings"],
        **original_config,
    )

    return new_config


def vision_conv_embeddings(idx):
    """
    The function helps in renaming vision convolution embedding layer weights.

    Args:
        idx: stage number in original model
    """
    convs = []
    convs.append(
        (
            f"vision_tower.convs.{idx}.proj.weight",
            f"model.vision_tower.convs.{idx}.conv.weight",
        )
    )
    convs.append(
        (
            f"vision_tower.convs.{idx}.proj.bias",
            f"model.vision_tower.convs.{idx}.conv.bias",
        )
    )
    convs.append(
        (
            f"vision_tower.convs.{idx}.norm.weight",
            f"model.vision_tower.convs.{idx}.norm.weight",
        )
    )
    convs.append(
        (
            f"vision_tower.convs.{idx}.norm.bias",
            f"model.vision_tower.convs.{idx}.norm.bias",
        )
    )
    return convs


def vision_spatial_block(stage_idx, block_idx):
    """
    The function helps in renaming vision spatial block layers weights.

    Args:
        idx: stage number in original model
        cnt: count of blocks in each stage
    """
    spatial_block = []
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.conv1.fn.dw.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.conv1.weight",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.conv1.fn.dw.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.conv1.bias",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.window_attn.norm.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.norm1.weight",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.window_attn.norm.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.norm1.bias",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.window_attn.fn.qkv.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.window_attn.qkv.weight",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.window_attn.fn.qkv.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.window_attn.qkv.bias",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.window_attn.fn.proj.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.window_attn.proj.weight",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.window_attn.fn.proj.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.window_attn.proj.bias",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.conv2.fn.dw.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.conv2.weight",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.conv2.fn.dw.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.conv2.bias",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.ffn.norm.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.norm2.weight",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.ffn.norm.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.norm2.bias",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.ffn.fn.net.fc1.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.ffn.fc1.weight",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.ffn.fn.net.fc1.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.ffn.fc1.bias",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.ffn.fn.net.fc2.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.ffn.fc2.weight",
        )
    )
    spatial_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.ffn.fn.net.fc2.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.spatial_block.ffn.fc2.bias",
        )
    )
    return spatial_block


def vision_channel_block(stage_idx, block_idx):
    """
    The function helps in renaming vision channel block layers weights.

    Args:
        idx: stage number in original model
        cnt: count of blocks in each stage
    """
    channel_block = []
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.conv1.fn.dw.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.conv1.weight",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.conv1.fn.dw.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.conv1.bias",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.channel_attn.norm.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.norm1.weight",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.channel_attn.norm.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.norm1.bias",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.channel_attn.fn.qkv.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.channel_attn.qkv.weight",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.channel_attn.fn.qkv.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.channel_attn.qkv.bias",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.channel_attn.fn.proj.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.channel_attn.proj.weight",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.channel_attn.fn.proj.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.channel_attn.proj.bias",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.conv2.fn.dw.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.conv2.weight",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.conv2.fn.dw.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.conv2.bias",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.ffn.norm.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.norm2.weight",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.ffn.norm.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.norm2.bias",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.ffn.fn.net.fc1.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.ffn.fc1.weight",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.ffn.fn.net.fc1.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.ffn.fc1.bias",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.ffn.fn.net.fc2.weight",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.ffn.fc2.weight",
        )
    )
    channel_block.append(
        (
            f"vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.ffn.fn.net.fc2.bias",
            f"model.vision_tower.blocks.{stage_idx}.{block_idx}.channel_block.ffn.fc2.bias",
        )
    )
    return channel_block


def multi_modal_projector():
    """
    Function helps in renaming final classification layer
    """
    projector = []
    projector.append(("image_projection", "model.multi_modal_projector.image_projection.weight"))
    projector.append(("image_proj_norm.weight", "model.multi_modal_projector.image_proj_norm.weight"))
    projector.append(("image_proj_norm.bias", "model.multi_modal_projector.image_proj_norm.bias"))
    projector.append(
        (
            "image_pos_embed.row_embeddings.weight",
            "model.multi_modal_projector.image_position_embed.row_embeddings.weight",
        )
    )
    projector.append(
        (
            "image_pos_embed.column_embeddings.weight",
            "model.multi_modal_projector.image_position_embed.column_embeddings.weight",
        )
    )
    projector.append(
        (
            "visual_temporal_embed.pos_idx_to_embed",
            "model.multi_modal_projector.visual_temporal_embed.pos_idx_to_embed",
        )
    )
    return projector


def language_model(state_dict):
    language_state_dict_keys = []
    for key in state_dict.keys():
        if key.startswith("language_model.model") and "lm_head" not in key:
            new_key = key.replace("language_model.model.", "model.language_model.")
            language_state_dict_keys.append((key, new_key))
    language_state_dict_keys.append(("language_model.lm_head.weight", "lm_head.weight"))
    return language_state_dict_keys


def convert_florence2_checkpoint(hf_model_id, pytorch_dump_folder, output_hub_path):
    """
    Function to convert the microsoft florence2 checkpoint to huggingface checkpoint
    """

    hf_config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, trust_remote_code=True, dtype=torch.float16, attn_implementation="eager"
    )
    hf_processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)
    huggingface_weights = OrderedDict()
    list_of_state_dict = []

    image_processor = hf_processor.image_processor

    tokenizer = hf_processor.tokenizer
    tokenizer.image_token = "<image>"
    tokenizer.add_tokens(AddedToken(tokenizer.image_token, special=True, normalized=False), special_tokens=True)
    tokenizer.image_token_id = tokenizer.encode(tokenizer.image_token, add_special_tokens=False)[0]
    tokenizer.extra_special_tokens = {"image_token": "<image>"}

    post_processor_config = {
        "ocr": {
            "pattern": r"(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>",
            "area_threshold": 0.0,
        },
        "phrase_grounding": {
            "banned_grounding_tokens": [
                "it",
                "I",
                "me",
                "mine",
                "you",
                "your",
                "yours",
                "he",
                "him",
                "his",
                "she",
                "her",
                "hers",
                "they",
                "them",
                "their",
                "theirs",
                "one",
                "oneself",
                "we",
                "us",
                "our",
                "ours",
                "you",
                "your",
                "yours",
                "they",
                "them",
                "their",
                "theirs",
                "mine",
                "yours",
                "his",
                "hers",
                "its",
                "ours",
                "yours",
                "theirs",
                "myself",
                "yourself",
                "himself",
                "herself",
                "itself",
                "ourselves",
                "yourselves",
                "themselves",
                "this",
                "that",
                "these",
                "those",
                "who",
                "whom",
                "whose",
                "which",
                "what",
                "who",
                "whom",
                "whose",
                "which",
                "that",
                "all",
                "another",
                "any",
                "anybody",
                "anyone",
                "anything",
                "each",
                "everybody",
                "everyone",
                "everything",
                "few",
                "many",
                "nobody",
                "none",
                "one",
                "several",
                "some",
                "somebody",
                "someone",
                "something",
                "each other",
                "one another",
                "myself",
                "yourself",
                "himself",
                "herself",
                "itself",
                "ourselves",
                "yourselves",
                "themselves",
                "the image",
                "image",
                "images",
                "the",
                "a",
                "an",
                "a group",
                "other objects",
                "lots",
                "a set",
            ],
        },
        "pure_text": {},
        "description_with_bboxes": {},
        "description_with_polygons": {},
        "polygons": {},
        "bboxes": {},
        "description_with_bboxes_or_polygons": {},
    }
    processor = Florence2Processor(
        image_processor=image_processor, tokenizer=tokenizer, post_processor_config=post_processor_config
    )

    vision_config = convert_config(hf_config.vision_config.__dict__)
    text_config = hf_config.text_config.__dict__
    config = Florence2Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=tokenizer.image_token_id,
        dtype=torch.float16,
    )

    for stage_idx in range(len(config.vision_config.embed_dim)):
        list_of_state_dict = list_of_state_dict + vision_conv_embeddings(stage_idx)
        for block_idx in range(config.vision_config.depths[stage_idx]):
            list_of_state_dict = list_of_state_dict + vision_spatial_block(stage_idx, block_idx)
            list_of_state_dict = list_of_state_dict + vision_channel_block(stage_idx, block_idx)

    original_weights = hf_model.state_dict()
    list_of_state_dict = list_of_state_dict + multi_modal_projector()
    list_of_state_dict = list_of_state_dict + language_model(original_weights)
    for i in range(len(list_of_state_dict)):
        if list_of_state_dict[i][0] == "image_projection":
            original_weights[list_of_state_dict[i][0]].transpose_(1, 0)
        huggingface_weights[list_of_state_dict[i][1]] = original_weights[list_of_state_dict[i][0]]

    model = Florence2ForConditionalGeneration(config)
    model.load_state_dict(huggingface_weights, strict=True, assign=True)
    model.tie_weights()
    # We add an image token so we resize the model and pad to 64 for performance reasons
    pad_shape = 64
    model.resize_token_embeddings(len(tokenizer), pad_shape)

    if pytorch_dump_folder:
        model.save_pretrained(pytorch_dump_folder)
        processor.save_pretrained(pytorch_dump_folder)

    if output_hub_path:
        model.push_to_hub(output_hub_path)
        processor.push_to_hub(output_hub_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model_id",
        default="microsoft/Florence-2-base",
        type=str,
        help="Name of the florence2 model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--output_hub_path",
        help="Location on the hub of the converted model",
    )

    args = parser.parse_args()
    convert_florence2_checkpoint(args.hf_model_id, args.pytorch_dump_folder_path, args.output_hub_path)
