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
from GroupViT.utils.config import load_config
from GroupViT.models import build_model

from transformers import GroupViTConfig, GroupViTModel


def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    q_proj, k_proj, v_proj = pt_attn_layer.in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = pt_attn_layer.in_proj_bias.chunk(3, dim=0)

    out_proj_weights = pt_attn_layer.out_proj.weight
    out_proj_bias = pt_attn_layer.out_proj.bias

    hf_attn_layer.q_proj.weight.data = q_proj
    hf_attn_layer.q_proj.bias.data = q_proj_bias

    hf_attn_layer.k_proj.weight.data = k_proj
    hf_attn_layer.k_proj.bias.data = k_proj_bias

    hf_attn_layer.v_proj.weight.data = v_proj
    hf_attn_layer.v_proj.bias.data = v_proj_bias

    hf_attn_layer.out_proj.weight = out_proj_weights
    hf_attn_layer.out_proj.bias = out_proj_bias


def copy_mlp(hf_mlp, pt_mlp):
    copy_linear(hf_mlp.fc1, pt_mlp.c_fc)
    copy_linear(hf_mlp.fc2, pt_mlp.c_proj)


def copy_linear(hf_linear, pt_linear):
    hf_linear.weight = pt_linear.weight
    hf_linear.bias = pt_linear.bias


def copy_linear_from_conv1d(hf_linear, pt_conv1d):
    assert hf_linear.weight.shape == pt_conv1d.weight.shape[:2]
    hf_linear.weight.data = pt_conv1d.weight.data.squeeze()
    hf_linear.bias = pt_conv1d.bias


def copy_bn(hf_bn, pt_bn):
    hf_bn.weight = pt_bn.weight
    hf_bn.bias = pt_bn.bias
    hf_bn.running_mean = pt_bn.running_mean
    hf_bn.running_var = pt_bn.running_var


def copy_layer(hf_layer, pt_layer):
    # copy layer norms
    copy_linear(hf_layer.layer_norm1, pt_layer.ln_1)
    copy_linear(hf_layer.layer_norm2, pt_layer.ln_2)

    # copy MLP
    copy_mlp(hf_layer.mlp, pt_layer.mlp)

    # copy attn
    copy_attn_layer(hf_layer.self_attn, pt_layer.attn)


def copy_layers(hf_layers, pt_layers):
    for hf_layer, pt_layer in zip(hf_layers, pt_layers):
        copy_layer(hf_layer, pt_layer)


def copy_projection(hf_projection, pt_projection):
    copy_linear_from_conv1d(hf_projection[0], pt_projection.linear_hidden[0])
    copy_bn(hf_projection[1], pt_projection.linear_hidden[1])
    copy_linear_from_conv1d(hf_projection[3], pt_projection.linear_out)


def copy_text_encoder(hf_encoder, pt_model):
    # copy  embeds
    hf_encoder.embeddings.token_embedding.weight = pt_model.token_embedding.weight
    hf_encoder.embeddings.position_embedding.weight.data = pt_model.positional_embedding

    # copy layer norm
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)

    # copy hidden layers
    copy_layers(hf_encoder.encoder.layers, pt_model.transformer.resblocks)


def copy_text_model_and_projection(hf_model, pt_model):
    # copy projection
    copy_projection(hf_model.text_projection, pt_model.text_projector)
    # copy text encoder
    copy_text_encoder(hf_model.text_model, pt_model.text_encoder)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys_for_vision_model(config):
    rename_keys = []
    for d, depth in enumerate(config.depths):
        for i in range(depth):
            rename_keys.append(
                (
                    f"img_encoder.layers.{d}.blocks.{i}.norm1.weight",
                    f"vision_model.encoder.layer.{d}.blocks.{i}.layernorm_before.weight",
                )
            )
            rename_keys.append(
                (
                    f"img_encoder.layers.{d}.blocks.{i}.norm1.bias",
                    f"vision_model.encoder.layer.{d}.blocks.{i}.layernorm_before.bias",
                )
            )
            rename_keys.append(
                (
                    f"img_encoder.layers.{d}.blocks.{i}.attn.proj.weight",
                    f"vision_model.encoder.layer.{d}.blocks.{i}.attention.output.dense.weight",
                )
            )
            rename_keys.append(
                (
                    f"img_encoder.layers.{d}.blocks.{i}.attn.proj.bias",
                    f"vision_model.encoder.layer.{d}.blocks.{i}.attention.output.dense.bias",
                )
            )
            rename_keys.append(
                (
                    f"img_encoder.layers.{d}.blocks.{i}.norm2.weight",
                    f"vision_model.encoder.layer.{d}.blocks.{i}.layernorm_after.weight",
                )
            )
            rename_keys.append(
                (
                    f"img_encoder.layers.{d}.blocks.{i}.norm2.bias",
                    f"vision_model.encoder.layer.{d}.blocks.{i}.layernorm_after.bias",
                )
            )
            rename_keys.append(
                (
                    f"img_encoder.layers.{d}.blocks.{i}.mlp.fc1.weight",
                    f"vision_model.encoder.layer.{d}.blocks.{i}.intermediate.dense.weight",
                )
            )
            rename_keys.append(
                (
                    f"img_encoder.layers.{d}.blocks.{i}.mlp.fc1.bias",
                    f"vision_model.encoder.layer.{d}.blocks.{i}.intermediate.dense.bias",
                )
            )
            rename_keys.append(
                (
                    f"img_encoder.layers.{d}.blocks.{i}.mlp.fc2.weight",
                    f"vision_model.encoder.layer.{d}.blocks.{i}.output.dense.weight",
                )
            )
            rename_keys.append(
                (
                    f"img_encoder.layers.{d}.blocks.{i}.mlp.fc2.bias",
                    f"vision_model.encoder.layer.{d}.blocks.{i}.output.dense.bias",
                )
            )

    # projection layer + position embeddings
    rename_keys.extend(
        [
            ("img_encoder.patch_embed.proj.weight", "vision_model.embeddings.patch_embeddings.weight"),
            ("img_encoder.patch_embed.proj.bias", "vision_model.embeddings.patch_embeddings.bias"),
            ("img_encoder.patch_embed.norm.weight", "vision_model.embeddings.layer_norm.weight"),
            ("img_encoder.patch_embed.norm.bias", "vision_model.embeddings.layer_norm.bias"),
            ("img_encoder.pos_embed", "vision_model.embeddings.position_embeddings"),
            ("img_encoder.norm.weight", "vision_model.layernorm.weight"),
            ("img_encoder.norm.bias", "vision_model.layernorm.bias"),
        ]
    )

    return rename_keys


def create_rename_keys_for_text_model(config):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        rename_keys.append(
            (f"text_encoder.layers.{i}.norm1.weight", f"text_model.encoder.layer.{i}.layernorm_before.weight")
        )


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v_for_vision_model(state_dict, config):
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


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def copy_vison_model_and_projection(hf_model, pt_model, config):
    # copy projection
    copy_projection(hf_model.visual_projection, pt_model.img_projector)

    # copy encoder
    rename_keys = create_rename_keys_for_vision_model(config)
    state_dict = pt_model.state_dict()
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v_for_vision_model(state_dict, config)


@torch.no_grad()
def convert_groupvit_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    config = GroupViTConfig()

    hf_model = GroupViTModel(config).eval()
    print(hf_model)

    pt_model = build_model(load_config("GroupViT/configs/group_vit_gcc_yfcc_30e.yml").model)
    if checkpoint_path.startswith("https:"):
        pt_model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_path, map_location="cpu")["model"])
    else:
        pt_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])
    pt_model = pt_model.eval()

    copy_text_model_and_projection(hf_model, pt_model)
    copy_vison_model_and_projection(hf_model, pt_model, config.vision_config)
    hf_model.logit_scale = pt_model.logit_scale

    input_ids = torch.arange(0, 77).unsqueeze(0)
    pixel_values = torch.randn(1, 3, 224, 224)

    hf_logits_per_image, hf_logits_per_text = hf_model(
        input_ids=input_ids, pixel_values=pixel_values, return_dict=True
    )[1:3]
    pt_logits_per_image, pt_logits_per_text = pt_model(pixel_values, input_ids)

    assert torch.allclose(hf_logits_per_image, pt_logits_per_image, atol=1e-3)
    assert torch.allclose(hf_logits_per_text, pt_logits_per_text, atol=1e-3)

    hf_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to GroupViT checkpoint")
    args = parser.parse_args()

    convert_groupvit_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path)
