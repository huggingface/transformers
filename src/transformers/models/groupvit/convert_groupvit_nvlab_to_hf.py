# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
import torch.nn.functional as F

from GroupViT.models import build_model
from GroupViT.utils.config import load_config
from transformers import GroupViTConfig, GroupViTModel


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys_for_vision_model(config, hf_model, pt_model):
    rename_keys = []

    # position embeddings
    rename_keys.extend(
        [
            ("img_encoder.pos_embed", "vision_model.embeddings.position_embeddings"),
        ]
    )

    for suffix in ["weight", "bias"]:
        # projection layer + position embeddings
        rename_keys.extend(
            [
                (
                    f"img_encoder.patch_embed.proj.{suffix}",
                    f"vision_model.embeddings.patch_embeddings.projection.{suffix}",
                ),
                (f"img_encoder.patch_embed.norm.{suffix}", f"vision_model.embeddings.layernorm.{suffix}"),
                (f"img_encoder.norm.{suffix}", f"vision_model.layernorm.{suffix}"),
            ]
        )

    for d, depth in enumerate(config.depths):
        if config.num_group_tokens[d] > 0:
            rename_keys.extend(
                [
                    (f"img_encoder.layers.{d}.group_token", f"vision_model.encoder.layer.{d}.group_token"),
                ]
            )
            for module_name in ["downsample", "group_projector"]:
                for name, _ in pt_model.named_parameters():
                    if name.startswith(f"img_encoder.layers.{d}.{module_name}"):
                        rename_keys.append((name, name.replace("img_encoder.layers.", "vision_model.encoder.layer.")))

        for i in range(depth):
            for suffix in ["weight", "bias"]:
                rename_keys.extend(
                    [
                        (
                            f"img_encoder.layers.{d}.blocks.{i}.norm1.{suffix}",
                            f"vision_model.encoder.layer.{d}.blocks.{i}.layernorm_before.{suffix}",
                        ),
                        (
                            f"img_encoder.layers.{d}.blocks.{i}.attn.proj.{suffix}",
                            f"vision_model.encoder.layer.{d}.blocks.{i}.attention.output.dense.{suffix}",
                        ),
                        (
                            f"img_encoder.layers.{d}.blocks.{i}.norm2.{suffix}",
                            f"vision_model.encoder.layer.{d}.blocks.{i}.layernorm_after.{suffix}",
                        ),
                        (
                            f"img_encoder.layers.{d}.blocks.{i}.mlp.fc1.{suffix}",
                            f"vision_model.encoder.layer.{d}.blocks.{i}.intermediate.dense.{suffix}",
                        ),
                        (
                            f"img_encoder.layers.{d}.blocks.{i}.mlp.fc2.{suffix}",
                            f"vision_model.encoder.layer.{d}.blocks.{i}.output.dense.{suffix}",
                        ),
                    ]
                )
    for suffix in ["weight", "bias"]:
        rename_keys.extend(
            [
                (f"img_projector.linear_hidden.0.{suffix}", f"visual_projection.0.{suffix}"),
                (f"img_projector.linear_hidden.1.{suffix}", f"visual_projection.1.{suffix}"),
                (f"img_projector.linear_out.{suffix}", f"visual_projection.3.{suffix}"),
            ]
        )
    for suffix in ["running_mean", "running_var", "num_batches_tracked"]:
        rename_keys.extend(
            [
                (f"img_projector.linear_hidden.1.{suffix}", f"visual_projection.1.{suffix}"),
            ]
        )

    return rename_keys


def create_rename_keys_for_text_model(config, hf_model, pt_model):
    rename_keys = []

    rename_keys.extend(
        [
            ("text_encoder.positional_embedding", "text_model.embeddings.position_embedding.weight"),
            ("text_encoder.token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
        ]
    )

    for i in range(config.num_hidden_layers):
        for suffix in ["weight", "bias"]:
            rename_keys.extend(
                [
                    (
                        f"text_encoder.transformer.resblocks.{i}.ln_1.{suffix}",
                        f"text_model.encoder.layers.{i}.layer_norm1.{suffix}",
                    ),
                    (
                        f"text_encoder.transformer.resblocks.{i}.ln_2.{suffix}",
                        f"text_model.encoder.layers.{i}.layer_norm2.{suffix}",
                    ),
                    (
                        f"text_encoder.transformer.resblocks.{i}.mlp.c_fc.{suffix}",
                        f"text_model.encoder.layers.{i}.mlp.fc1.{suffix}",
                    ),
                    (
                        f"text_encoder.transformer.resblocks.{i}.mlp.c_proj.{suffix}",
                        f"text_model.encoder.layers.{i}.mlp.fc2.{suffix}",
                    ),
                    (
                        f"text_encoder.transformer.resblocks.{i}.attn.out_proj.{suffix}",
                        f"text_model.encoder.layers.{i}.self_attn.out_proj.{suffix}",
                    ),
                ]
            )

    for suffix in ["weight", "bias"]:
        rename_keys.extend(
            [
                (f"text_encoder.ln_final.{suffix}", f"text_model.final_layer_norm.{suffix}"),
                (f"text_projector.linear_hidden.0.{suffix}", f"text_projection.0.{suffix}"),
                (f"text_projector.linear_hidden.1.{suffix}", f"text_projection.1.{suffix}"),
                (f"text_projector.linear_out.{suffix}", f"text_projection.3.{suffix}"),
            ]
        )
    for suffix in ["running_mean", "running_var", "num_batches_tracked"]:
        rename_keys.extend(
            [
                (f"text_projector.linear_hidden.1.{suffix}", f"text_projection.1.{suffix}"),
            ]
        )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def format_shape_for_vision_model(state_dict, config):
    for d, depth in enumerate(config.depths):
        for i in range(depth):
            # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
            in_proj_weight = state_dict.pop(f"img_encoder.layers.{d}.blocks.{i}.attn.qkv.weight")
            in_proj_bias = state_dict.pop(f"img_encoder.layers.{d}.blocks.{i}.attn.qkv.bias")
            # next, add query, keys and values (in that order) to the state dict
            state_dict[f"vision_model.encoder.layer.{d}.blocks.{i}.attention.attention.query.weight"] = in_proj_weight[
                : config.hidden_size, :
            ]
            state_dict[f"vision_model.encoder.layer.{d}.blocks.{i}.attention.attention.query.bias"] = in_proj_bias[
                : config.hidden_size
            ]
            state_dict[f"vision_model.encoder.layer.{d}.blocks.{i}.attention.attention.key.weight"] = in_proj_weight[
                config.hidden_size : config.hidden_size * 2, :
            ]
            state_dict[f"vision_model.encoder.layer.{d}.blocks.{i}.attention.attention.key.bias"] = in_proj_bias[
                config.hidden_size : config.hidden_size * 2
            ]
            state_dict[f"vision_model.encoder.layer.{d}.blocks.{i}.attention.attention.value.weight"] = in_proj_weight[
                -config.hidden_size :, :
            ]
            state_dict[f"vision_model.encoder.layer.{d}.blocks.{i}.attention.attention.value.bias"] = in_proj_bias[
                -config.hidden_size :
            ]
    state_dict["visual_projection.0.weight"] = state_dict["visual_projection.0.weight"].squeeze()
    state_dict["visual_projection.3.weight"] = state_dict["visual_projection.3.weight"].squeeze()


def format_shape_for_text_model(state_dict, config):
    for i in range(config.num_hidden_layers):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"text_encoder.transformer.resblocks.{i}.attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"text_encoder.transformer.resblocks.{i}.attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"text_model.encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"text_model.encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"text_model.encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"text_model.encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"text_model.encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-config.hidden_size :, :]
        state_dict[f"text_model.encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-config.hidden_size :]
    state_dict["text_projection.0.weight"] = state_dict["text_projection.0.weight"].squeeze()
    state_dict["text_projection.3.weight"] = state_dict["text_projection.3.weight"].squeeze()


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


@torch.no_grad()
def pt_model_forward(pt_model, pixel_values, input_ids):
    image_x = pt_model.encode_image(pixel_values)
    image_x = F.normalize(image_x, dim=-1)

    text_x = pt_model.encode_text(input_ids)
    text_x = F.normalize(text_x, dim=-1)

    logits_per_img = image_x @ text_x.t()
    logits_per_text = text_x @ image_x.t()

    logits_per_img = logits_per_img * pt_model.logit_scale.exp()
    logits_per_text = logits_per_text * pt_model.logit_scale.exp()

    return logits_per_img, logits_per_text


@torch.no_grad()
def convert_groupvit_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    config = GroupViTConfig()

    hf_model = GroupViTModel(config).eval()

    pt_model = build_model(load_config("GroupViT/configs/group_vit_gcc_yfcc_30e.yml").model)
    if checkpoint_path.startswith("https:"):
        pt_model.load_state_dict(
            torch.hub.load_state_dict_from_url(checkpoint_path, map_location="cpu")["model"], strict=True
        )
    else:
        pt_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"], strict=True)
    pt_model = pt_model.eval()

    state_dict = pt_model.state_dict().copy()
    rename_keys = create_rename_keys_for_vision_model(config.vision_config, hf_model, pt_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    format_shape_for_vision_model(state_dict, config.vision_config)

    rename_keys = create_rename_keys_for_text_model(config.text_config, hf_model, pt_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    format_shape_for_text_model(state_dict, config.text_config)
    missing_keys, unexpected_keys = hf_model.load_state_dict(state_dict, strict=False)
    assert missing_keys == ["text_model.embeddings.position_ids"]
    assert unexpected_keys == ["multi_label_logit_scale"]

    input_ids = torch.arange(0, 77).unsqueeze(0)
    pixel_values = torch.randn(1, 3, 224, 224)

    hf_logits_per_image, hf_logits_per_text = hf_model(
        input_ids=input_ids, pixel_values=pixel_values, return_dict=True
    )[:2]
    pt_logits_per_image, pt_logits_per_text = pt_model_forward(pt_model, pixel_values, input_ids)

    assert torch.allclose(hf_logits_per_image, pt_logits_per_image, atol=1e-3)
    assert torch.allclose(hf_logits_per_text, pt_logits_per_text, atol=1e-3)

    hf_model.save_pretrained(pytorch_dump_folder_path)
    print("Saved model to", pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to GroupViT checkpoint")
    args = parser.parse_args()

    convert_groupvit_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path)
