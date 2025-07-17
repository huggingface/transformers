# coding=utf-8
# Copyright 2022 The OFA-Sys Team Authors and The HuggingFace Team. All rights reserved.
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

from transformers import ChineseCLIPConfig, ChineseCLIPModel


def copy_attn_layer(hf_attn_layer, pt_weights, prefix):
    q_proj, k_proj, v_proj = pt_weights[f"{prefix}.in_proj_weight"].chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = pt_weights[f"{prefix}.in_proj_bias"].chunk(3, dim=0)

    out_proj_weights = pt_weights[f"{prefix}.out_proj.weight"]
    out_proj_bias = pt_weights[f"{prefix}.out_proj.bias"]

    hf_attn_layer.q_proj.weight.data = q_proj
    hf_attn_layer.q_proj.bias.data = q_proj_bias

    hf_attn_layer.k_proj.weight.data = k_proj
    hf_attn_layer.k_proj.bias.data = k_proj_bias

    hf_attn_layer.v_proj.weight.data = v_proj
    hf_attn_layer.v_proj.bias.data = v_proj_bias

    hf_attn_layer.out_proj.weight.data = out_proj_weights
    hf_attn_layer.out_proj.bias.data = out_proj_bias


def copy_mlp(hf_mlp, pt_weights, prefix):
    copy_linear(hf_mlp.fc1, pt_weights, f"{prefix}.c_fc")
    copy_linear(hf_mlp.fc2, pt_weights, f"{prefix}.c_proj")


def copy_linear(hf_linear, pt_weights, prefix):
    hf_linear.weight.data = pt_weights[f"{prefix}.weight"].data
    hf_linear.bias.data = pt_weights[f"{prefix}.bias"].data


def copy_layer(hf_layer, pt_weights, prefix):
    # copy layer norms
    copy_linear(hf_layer.layer_norm1, pt_weights, f"{prefix}.ln_1")
    copy_linear(hf_layer.layer_norm2, pt_weights, f"{prefix}.ln_2")

    # copy MLP
    copy_mlp(hf_layer.mlp, pt_weights, f"{prefix}.mlp")

    # copy attn
    copy_attn_layer(hf_layer.self_attn, pt_weights, f"{prefix}.attn")


def copy_layers(hf_layers, pt_weights, prefix):
    for layer_id, hf_layer in enumerate(hf_layers):
        copy_layer(hf_layer, pt_weights, f"{prefix}.{layer_id}")


def copy_text_model_and_projection(hf_model, pt_weights):
    # copy projection
    hf_model.text_projection.weight.data = pt_weights["text_projection"].data.T

    # copy text encoder
    for name, param in hf_model.text_model.named_parameters():
        param.data = pt_weights[f"bert.{name}"].data


def copy_vision_model_and_projection(hf_model, pt_weights):
    # copy projection
    hf_model.visual_projection.weight.data = pt_weights["visual.proj"].data.T

    # copy layer norms
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_weights, "visual.ln_pre")
    copy_linear(hf_model.vision_model.post_layernorm, pt_weights, "visual.ln_post")

    # copy embeddings
    hf_model.vision_model.embeddings.patch_embedding.weight.data = pt_weights["visual.conv1.weight"].data
    hf_model.vision_model.embeddings.class_embedding.data = pt_weights["visual.class_embedding"].data
    hf_model.vision_model.embeddings.position_embedding.weight.data = pt_weights["visual.positional_embedding"].data

    # copy encoder
    copy_layers(hf_model.vision_model.encoder.layers, pt_weights, "visual.transformer.resblocks")


@torch.no_grad()
def convert_chinese_clip_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """

    assert config_path is not None, "Please specify the ChineseCLIP model config of the corresponding model size."
    config = ChineseCLIPConfig.from_pretrained(config_path)

    hf_model = ChineseCLIPModel(config).eval()

    pt_weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]
    pt_weights = {(name[7:] if name.startswith("module.") else name): value for name, value in pt_weights.items()}

    copy_text_model_and_projection(hf_model, pt_weights)
    copy_vision_model_and_projection(hf_model, pt_weights)
    hf_model.logit_scale.data = pt_weights["logit_scale"].data

    hf_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output folder storing converted hf PyTorch model.",
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to original github format ChineseCLIP checkpoint."
    )
    parser.add_argument(
        "--config_path", default=None, required=True, type=str, help="Path to hf config.json of model to convert."
    )
    args = parser.parse_args()

    convert_chinese_clip_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
    print("The conversion is finished!")
