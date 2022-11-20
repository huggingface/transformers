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
"""Convert OWL-ViT checkpoints from the original repository. URL:
https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit"""

import argparse
import collections

import torch
import torch.nn as nn

import jax
import jax.numpy as jnp
from clip.model import CLIP
from flax.training import checkpoints
from huggingface_hub import Repository
from transformers import (
    CLIPTokenizer,
    OwlViTConfig,
    OwlViTFeatureExtractor,
    OwlViTForObjectDetection,
    OwlViTModel,
    OwlViTProcessor,
)


CONFIGS = {
    "vit_b32": dict(
        embed_dim=512,
        image_resolution=768,
        context_length=16,
        vocab_size=49408,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    ),
    "vit_b16": dict(
        embed_dim=512,
        image_resolution=768,
        context_length=16,
        vocab_size=49408,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=16,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    ),
    "vit_l14": dict(
        embed_dim=768,
        image_resolution=840,
        context_length=16,
        vocab_size=49408,
        vision_layers=24,
        vision_width=1024,
        vision_patch_size=14,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=12,
    ),
}


def flatten_nested_dict(params, parent_key="", sep="/"):
    items = []

    for k, v in params.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def to_f32(params):
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, params)


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
    hf_model.text_projection.weight.data = pt_model.text_projection.data.T

    # copy text encoder
    copy_encoder(hf_model.text_model, pt_model)


def copy_vision_model_and_projection(hf_model, pt_model):
    # copy projection
    hf_model.visual_projection.weight.data = pt_model.visual.proj.data.T

    # copy layer norms
    copy_linear(hf_model.vision_model.pre_layernorm, pt_model.visual.ln_pre)
    copy_linear(hf_model.vision_model.post_layernorm, pt_model.visual.ln_post)

    # copy embeds
    hf_model.vision_model.embeddings.patch_embedding.weight.data = pt_model.visual.conv1.weight.data
    hf_model.vision_model.embeddings.class_embedding = pt_model.visual.class_embedding
    hf_model.vision_model.embeddings.position_embedding.weight.data = pt_model.visual.positional_embedding.data

    # copy encoder
    copy_layers(hf_model.vision_model.encoder.layers, pt_model.visual.transformer.resblocks)


def copy_class_merge_token(hf_model, flax_params):
    flax_class_token_params = flatten_nested_dict(flax_params["backbone"]["merged_class_token"])

    weight = torch.from_numpy(flax_class_token_params["scale"])
    bias = torch.from_numpy(flax_class_token_params["bias"])
    hf_model.layer_norm.weight = nn.Parameter(weight)
    hf_model.layer_norm.bias = nn.Parameter(bias)


def copy_class_box_heads(hf_model, flax_params):
    pt_params = hf_model.state_dict()
    new_params = {}

    # Rename class prediction head flax params to pytorch HF
    flax_class_params = flatten_nested_dict(flax_params["class_head"])

    for flax_key, v in flax_class_params.items():
        torch_key = flax_key.replace("/", ".")
        torch_key = torch_key.replace(".kernel", ".weight")
        torch_key = torch_key.replace("Dense_0", "dense0")
        torch_key = "class_head." + torch_key

        if "weight" in torch_key and v.ndim == 2:
            v = v.T

        new_params[torch_key] = nn.Parameter(torch.from_numpy(v))

    # Rename box prediction box flax params to pytorch HF
    flax_box_params = flatten_nested_dict(flax_params["obj_box_head"])

    for flax_key, v in flax_box_params.items():
        torch_key = flax_key.replace("/", ".")
        torch_key = torch_key.replace(".kernel", ".weight")
        torch_key = torch_key.replace("_", "").lower()
        torch_key = "box_head." + torch_key

        if "weight" in torch_key and v.ndim == 2:
            v = v.T

        new_params[torch_key] = nn.Parameter(torch.from_numpy(v))

    # Copy flax params to PyTorch params
    for name, param in new_params.items():
        if name in pt_params.keys():
            pt_params[name].copy_(param)


def copy_flax_attn_params(hf_backbone, flax_attn_params):
    for k, v in flax_attn_params.items():
        if k.startswith("transformer"):
            torch_key = k.replace("transformer.resblocks", "text_model.encoder.layers")
        else:
            torch_key = k.replace("visual.transformer.resblocks", "vision_model.encoder.layers")

        torch_key = torch_key.replace("attn", "self_attn")
        torch_key = torch_key.replace("key", "k_proj")
        torch_key = torch_key.replace("value", "v_proj")
        torch_key = torch_key.replace("query", "q_proj")
        torch_key = torch_key.replace("out", "out_proj")

        if "bias" in torch_key and v.ndim == 2:
            shape = v.shape[0] * v.shape[1]
            v = v.reshape(shape)

        if "weight" in torch_key and "out" in torch_key:
            shape = (v.shape[0] * v.shape[1], v.shape[2])
            v = v.reshape(shape).T

        if "weight" in torch_key and "out" not in torch_key:
            shape = (v.shape[0], v.shape[1] * v.shape[2])
            v = v.reshape(shape).T

        # Copy flax CLIP attn params to HF PyTorch params
        v = torch.from_numpy(v)
        hf_backbone.state_dict()[torch_key].copy_(v)


def _convert_attn_layers(params):
    new_params = {}
    processed_attn_layers = []

    for k, v in params.items():
        if "attn." in k:
            base = k[: k.rindex("attn.") + 5]
            if base in processed_attn_layers:
                continue

            processed_attn_layers.append(base)
            dim = params[base + "out.weight"].shape[-1]
            new_params[base + "out_proj.weight"] = params[base + "out.weight"].reshape(dim, dim).T
            new_params[base + "out_proj.bias"] = params[base + "out.bias"]
        else:
            new_params[k] = v
    return new_params


def convert_clip_backbone(flax_params, torch_config):
    torch_model = CLIP(**torch_config)
    torch_model.eval()
    torch_clip_params = torch_model.state_dict()

    flax_clip_params = flatten_nested_dict(flax_params["backbone"]["clip"])
    new_torch_params = {}

    for flax_key, v in flax_clip_params.items():
        torch_key = flax_key.replace("/", ".")
        torch_key = torch_key.replace("text.token_embedding.embedding", "token_embedding.kernel")

        if (
            torch_key.startswith("text.transformer")
            or torch_key.startswith("text.text_projection")
            or torch_key.startswith("text.ln_final")
            or torch_key.startswith("text.positional_embedding")
        ):
            torch_key = torch_key[5:]

        torch_key = torch_key.replace("text_projection.kernel", "text_projection")
        torch_key = torch_key.replace("visual.proj.kernel", "visual.proj")
        torch_key = torch_key.replace(".scale", ".weight")
        torch_key = torch_key.replace(".kernel", ".weight")

        if "conv" in torch_key or "downsample.0.weight" in torch_key:
            v = v.transpose(3, 2, 0, 1)

        elif "weight" in torch_key and v.ndim == 2 and "embedding" not in torch_key:
            # Fully connected layers are transposed, embeddings are not
            v = v.T

        new_torch_params[torch_key] = v

    attn_params = _convert_attn_layers(new_torch_params)
    new_torch_params.update(attn_params)
    attn_params = {}

    # Copy flax CLIP backbone params to PyTorch params
    for name, param in new_torch_params.items():
        if name in torch_clip_params.keys():

            new_param = torch.from_numpy(new_torch_params[name])
            torch_clip_params[name].copy_(new_param)
        else:
            attn_params[name] = param

    return torch_clip_params, torch_model, attn_params


@torch.no_grad()
def convert_owlvit_checkpoint(pt_backbone, flax_params, attn_params, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    repo = Repository(pytorch_dump_folder_path, clone_from=f"google/{pytorch_dump_folder_path}")
    repo.git_pull()

    if config_path is not None:
        config = OwlViTConfig.from_pretrained(config_path)
    else:
        config = OwlViTConfig()

    hf_backbone = OwlViTModel(config).eval()
    hf_model = OwlViTForObjectDetection(config).eval()

    copy_text_model_and_projection(hf_backbone, pt_backbone)
    copy_vision_model_and_projection(hf_backbone, pt_backbone)
    hf_backbone.logit_scale = pt_backbone.logit_scale
    copy_flax_attn_params(hf_backbone, attn_params)

    hf_model.owlvit = hf_backbone
    copy_class_merge_token(hf_model, flax_params)
    copy_class_box_heads(hf_model, flax_params)

    # Save HF model
    hf_model.save_pretrained(repo.local_dir)

    # Initialize feature extractor
    feature_extractor = OwlViTFeatureExtractor(
        size=config.vision_config.image_size, crop_size=config.vision_config.image_size
    )
    # Initialize tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", pad_token="!", model_max_length=16)

    # Initialize processor
    processor = OwlViTProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    feature_extractor.save_pretrained(repo.local_dir)
    processor.save_pretrained(repo.local_dir)

    repo.git_add()
    repo.git_commit("Upload model and processor")
    repo.git_push()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--owlvit_version",
        default=None,
        type=str,
        required=True,
        help="OWL-ViT model name [clip_b16, clip_b32, clip_l14].",
    )
    parser.add_argument(
        "--owlvit_checkpoint", default=None, type=str, required=True, help="Path to flax model checkpoint."
    )
    parser.add_argument("--hf_config", default=None, type=str, required=True, help="Path to HF model config.")
    parser.add_argument(
        "--pytorch_dump_folder_path", default="hf_model", type=str, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()

    # Initialize PyToch clip model
    model_name = args.owlvit_version
    if model_name == "clip_b16":
        torch_config = CONFIGS["vit_b16"]
    elif model_name == "clip_b32":
        torch_config = CONFIGS["vit_b32"]
    elif model_name == "clip_l14":
        torch_config = CONFIGS["vit_l14"]

    # Load from checkpoint and convert params to float-32
    variables = checkpoints.restore_checkpoint(args.owlvit_checkpoint, target=None)["optimizer"]["target"]
    flax_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, variables)
    del variables

    # Convert CLIP backbone
    pt_backbone_params, clip_pt, attn_params = convert_clip_backbone(flax_params, torch_config)

    convert_owlvit_checkpoint(clip_pt, flax_params, attn_params, args.pytorch_dump_folder_path, args.hf_config)
