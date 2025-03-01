# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert MLCD checkpoints from the original repository.

Usage:
    python convert_mlcd_weights_to_hf.py \
        --pytorch_dump_folder_path mlcd-vit-bigG-patch14-336 \
        --checkpoint_path MLCD_ViT_bigG_14_336px_pytorch.pt \
        --config_path mlcd-vit-bigG-patch14-336/config.json

URL: https://github.com/deepglint/unicom/tree/main"""

import argparse

import requests
import torch
from clip.clip import _transform
from PIL import Image

from transformers import (
    CLIPImageProcessor,
    MLCDVisionConfig,
    MLCDVisionModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .original_vit_rope2d import RoPE2d_ViT_bigG_14_1024


def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    # self.in_proj = nn.Linear(dim, dim * 3, bias=True)
    # self.out_proj = nn.Linear(dim, dim)

    q_proj, k_proj, v_proj = pt_attn_layer.in_proj.weight.data.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = pt_attn_layer.in_proj.bias.data.chunk(3, dim=0)

    out_proj_weights = pt_attn_layer.out_proj.weight
    out_proj_bias = pt_attn_layer.out_proj.bias

    hf_attn_layer.q_proj.weight.data = q_proj
    hf_attn_layer.q_proj.bias.data = q_proj_bias

    hf_attn_layer.k_proj.weight.data = k_proj
    hf_attn_layer.k_proj.bias.data = k_proj_bias

    hf_attn_layer.v_proj.weight.data = v_proj
    hf_attn_layer.v_proj.bias.data = v_proj_bias

    hf_attn_layer.out_proj.weight.data = out_proj_weights.data
    hf_attn_layer.out_proj.bias.data = out_proj_bias.data


def copy_mlp(hf_mlp, pt_mlp):
    copy_linear(hf_mlp.fc1, pt_mlp.c_fc)
    copy_linear(hf_mlp.fc2, pt_mlp.c_proj)


def copy_linear(hf_linear, pt_linear):
    hf_linear.weight.data = pt_linear.weight.data
    hf_linear.bias.data = pt_linear.bias.data


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


def copy_encoder(hf_encoder, pt_model):
    # copy  embeds
    hf_encoder.embeddings.token_embedding.weight = pt_model.token_embedding.weight
    hf_encoder.embeddings.position_embedding.weight.data = pt_model.positional_embedding

    # copy layer norm
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)

    # copy hidden layers
    copy_layers(hf_encoder.encoder.layers, pt_model.transformer.resblocks)


def copy_text_model_and_projection(hf_model, pt_model):
    # copy projection
    hf_model.text_projection.weight.data = pt_model.text_projection.data.T.contiguous()

    # copy text encoder
    copy_encoder(hf_model.text_model, pt_model)


def copy_vison_model(hf_model, pt_model):
    # copy projection
    # hf_model.visual_projection.weight.data = pt_model.visual.proj.data.T.contiguous()

    # copy layer norms
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_model.ln_pre)
    copy_linear(hf_model.vision_model.post_layernorm, pt_model.ln_post)

    # copy embeds
    hf_model.vision_model.embeddings.patch_embedding.weight.data = pt_model.conv1.weight.data
    hf_model.vision_model.embeddings.class_embedding = pt_model.class_embedding

    # copy encoder
    copy_layers(hf_model.vision_model.encoder.layers, pt_model.transformer.resblocks)

    # copy rope2d
    hf_model.vision_model.vision_rotary_embedding = pt_model.vision_rotary_embedding
    hf_model.vision_model.class_pos_emb.data = pt_model.class_pos_emb.data


@torch.no_grad()
def convert_clip_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """

    if config_path is not None:
        config = MLCDVisionConfig.from_pretrained(config_path)
    else:
        config = MLCDVisionConfig(
            hidden_size=1664,
            intermediate_size=8192,
            projection_dim=1024,
            num_hidden_layers=48,
            num_attention_heads=16,
            num_channels=3,
            image_size=336,
            patch_size=14,
            hidden_act="gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
        )

    hf_model = MLCDVisionModel(config).eval()

    state_dict = torch.load(checkpoint_path, "cpu")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    if "positional_embedding" in state_dict:
        state_dict.pop("positional_embedding")

    model_native = RoPE2d_ViT_bigG_14_1024().eval()
    model_native.load_state_dict(state_dict, strict=False)
    model_native = model_native.eval()

    copy_vison_model(hf_model, model_native)
    hf_model.save_pretrained(pytorch_dump_folder_path)

    mlcd_image_processor = CLIPImageProcessor(
        do_center_crop=True,
        do_normalize=True,
        do_resize=True,
        feature_extractor_type="CLIPFeatureExtractor",
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        resample=3,
        size=config.image_size,
        crop_size=config.image_size,
    )
    mlcd_image_processor.save_pretrained(pytorch_dump_folder_path)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    hf_image = mlcd_image_processor(image, return_tensors="pt")

    result = hf_model(hf_image.pixel_values, output_hidden_states=True)
    result: BaseModelOutputWithPooling

    # naive
    pt_image = _transform(config.image_size)(image)
    pt_image = pt_image.unsqueeze(0)
    list_hidden_states = model_native.forward_hidden_states(pt_image)

    assert torch.allclose(pt_image, hf_image.pixel_values, atol=1e-4)
    assert torch.allclose(result.hidden_states[-1], list_hidden_states[-1], atol=1e-4)

    # cosine
    for i in range(49):
        cosine = torch.nn.functional.cosine_similarity(
            result.hidden_states[i][:, 0].reshape(-1), list_hidden_states[i][:, 0].reshape(-1), dim=-1
        )
        cosine = cosine.item()
        assert cosine > 0.9999, f"cosine: {cosine} at layer {i}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()

    convert_clip_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
