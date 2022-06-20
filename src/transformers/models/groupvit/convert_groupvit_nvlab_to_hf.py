# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

"""
Convert GroupViT checkpoints from the original repository.

URL: https://github.com/NVlabs/GroupViT
"""

import argparse

import torch
import torch.nn.functional as F

from regex import E
from transformers import GroupViTConfig, GroupViTModel


def rename_key(name):
    # vision encoder
    if "img_encoder.pos_embed" in name:
        name = name.replace("img_encoder.pos_embed", "vision_model.embeddings.position_embeddings")
    if "img_encoder.patch_embed.proj" in name:
        name = name.replace("img_encoder.patch_embed.proj", "vision_model.embeddings.patch_embeddings.projection")
    if "img_encoder.patch_embed.norm" in name:
        name = name.replace("img_encoder.patch_embed.norm", "vision_model.embeddings.patch_embeddings.layernorm")
    if "img_encoder.layers" in name:
        name = name.replace("img_encoder.layers", "vision_model.encoder.stages")
    if "blocks" in name and "res" not in name:
        name = name.replace("blocks", "layers")
    if "attn" in name and "pre_assign" not in name:
        name = name.replace("attn", "self_attn")
    if "norm1" in name:
        name = name.replace("norm1", "layer_norm1")
    if "norm2" in name:
        name = name.replace("norm2", "layer_norm2")
    if "img_encoder.norm" in name:
        name = name.replace("img_encoder.norm", "vision_model.layernorm")
    # text encoder
    if "text_encoder.token_embedding" in name:
        name = name.replace("text_encoder.token_embedding", "text_model.embeddings.token_embedding")
    if "text_encoder.positional_embedding" in name:
        name = name.replace("text_encoder.positional_embedding", "text_model.embeddings.position_embedding.weight")
    if "text_encoder.transformer.resblocks." in name:
        name = name.replace("text_encoder.transformer.resblocks.", "text_model.encoder.layers.")
    if "ln_1" in name:
        name = name.replace("ln_1", "layer_norm1")
    if "ln_2" in name:
        name = name.replace("ln_2", "layer_norm2")
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    if "text_encoder" in name:
        name = name.replace("text_encoder", "text_model")
    if "ln_final" in name:
        name = name.replace("ln_final", "final_layer_norm")
    # projection layers
    if "img_projector.linear_hidden." in name:
        name = name.replace("img_projector.linear_hidden.", "visual_projection.")
    if "img_projector.linear_out." in name:
        name = name.replace("img_projector.linear_out.", "visual_projection.3.")
    if "text_projector.linear_hidden" in name:
        name = name.replace("text_projector.linear_hidden", "text_projection")
    if "text_projector.linear_out" in name:
        name = name.replace("text_projector.linear_out", "text_projection.3")

    return name


def convert_state_dict(orig_state_dict):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        # attention requires special treatment
        if "qkv" in key:
            pass
        else:
            new_name = rename_key(key)
            # squeeze if necessary
            if (
                "text_projection.0" in new_name
                or "text_projection.3" in new_name
                or "visual_projection.0" in new_name
                or "visual_projection.3" in new_name
            ):
                orig_state_dict[new_name] = val.squeeze_()
            else:
                orig_state_dict[new_name] = val

    return orig_state_dict


@torch.no_grad()
def convert_groupvit_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to the Transformers design.
    """
    config = GroupViTConfig()
    model = GroupViTModel(config).eval()

    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    new_state_dict = convert_state_dict(state_dict)
    del new_state_dict["multi_label_logit_scale"]
    model.load_state_dict(new_state_dict)

    expected_logits = torch.tensor([])

    model.save_pretrained(pytorch_dump_folder_path)
    print("Saved model to", pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to GroupViT checkpoint")
    args = parser.parse_args()

    convert_groupvit_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path)
