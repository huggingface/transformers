import os
import json
from typing import Any, Mapping, Optional
import argparse
import collections
from absl import logging

import flax
import jax
import jax.numpy as jnp
import numpy as np
import torch

import models
from clip_model import CLIP, OwlViTClassPredictor, OwlViTBoxPredictor, OwlViTImageTextEmbedder
from PIL import Image
from configs import clip_b16, clip_b32, clip_l14

PyTree = Any
CONFIGS = {
    'vit_b32': dict(embed_dim=512,
                    image_resolution=224,
                    context_length=16,
                    vocab_size=49408,
                    vision_layers=12,
                    vision_width=768,
                    vision_patch_size=32,
                    transformer_width=512,
                    transformer_heads=8,
                    transformer_layers=12),
    'vit_b16': dict(embed_dim=512,
                    image_resolution=224,
                    context_length=16,
                    vocab_size=49408,
                    vision_layers=12,
                    vision_width=768,
                    vision_patch_size=16,
                    transformer_width=512,
                    transformer_heads=8,
                    transformer_layers=12),
    'vit_l14': dict(embed_dim=768,
                    image_resolution=224,
                    context_length=16,
                    vocab_size=49408,
                    vision_layers=24,
                    vision_width=1024,
                    vision_patch_size=14,
                    transformer_width=768,
                    transformer_heads=12,
                    transformer_layers=12),
}


def flatten_nested_dict(params, parent_key='', sep='/'):
    items = []

    for k, v in params.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def to_f32(params):
    return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, params)


def _convert_attn_layers(params):
    new_params = {}
    processed_attn_layers = []

    for k, v in params.items():
        if 'attn.' in k:
            base = k[:k.rindex('attn.')+5]
            if base in processed_attn_layers:
                continue

            processed_attn_layers.append(base)
            dim = params[base + 'out.weight'].shape[-1]
            new_params[base + 'out_proj.weight'] = params[base + 'out.weight'].reshape(dim, dim).T
            new_params[base + 'out_proj.bias'] = params[base + 'out.bias']
        else:
            new_params[k] = v
    return new_params


def convert_clip_backbone(flax_params, torch_config):
    torch_model = CLIP(**torch_config)
    torch_clip_params = torch_model.state_dict()

    flax_clip_params = flatten_nested_dict(flax_params["backbone"]["clip"])
    new_torch_params = {}

    for flax_key, v in flax_clip_params.items():
        torch_key = flax_key.replace("/", ".")
        torch_key = torch_key.replace("text.token_embedding.embedding", "token_embedding.kernel")

        if (torch_key.startswith("text.transformer") or
            torch_key.startswith("text.text_projection") or
            torch_key.startswith("text.ln_final") or
            torch_key.startswith("text.positional_embedding")):
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

    # Copy flax CLIP backbone params to PyTorch params
    for name, param in new_torch_params.items():
        if name in torch_clip_params.keys():
            new_param = torch.from_numpy(new_torch_params[name])
            torch_clip_params[name].copy_(new_param)

    return torch_clip_params, torch_model


def convert_embedder(clip, flax_params, flax_config, torch_config):
    torch_model = OwlViTImageTextEmbedder(
        merge_class_token=flax_config.model.body.merge_class_token, 
        vision_width=torch_config["vision_width"],
        backbone=clip
    )
    torch_params = torch_model.state_dict()

    new_class_token_params = {}
    flax_class_token_params = flatten_nested_dict(flax_params["backbone"]["merged_class_token"])

    for flax_key, v in flax_class_token_params.items():
        torch_key = flax_key.replace("bias", "layer_norm.bias")
        torch_key = flax_key.replace("scale", "layer_norm.weight")
        new_class_token_params[torch_key] = v

    # Copy flax params to PyTorch params
    for name, param in new_class_token_params.items():
        if name in torch_params.keys():
            new_param = torch.from_numpy(new_class_token_params[name])
            torch_params[name].copy_(new_param)

    return torch_params
 

def convert_class_box_heads(flax_params, torch_config):
    # Initialize PyToch class head
    torch_model = OwlViTClassPredictor(out_dim=torch_config["embed_dim"], query_dim=torch_config["vision_width"])
    torch_class_params = torch_model.state_dict()

    # Convert flax params to pytorch
    new_class_head_params = {}
    flax_class_params = flatten_nested_dict(flax_params["class_head"])

    for flax_key, v in flax_class_params.items():
        torch_key = flax_key.replace("/", ".")
        torch_key = torch_key.replace(".kernel", ".weight")
        torch_key = torch_key.replace("Dense_0", "dense0")

        if "weight" in torch_key and v.ndim == 2:
            v = v.T

        new_class_head_params[torch_key] = v

    # Copy flax class head params to PyTorch params
    for name, param in new_class_head_params.items():
        if name in torch_class_params.keys():
            new_param = torch.from_numpy(new_class_head_params[name])
            torch_class_params[name].copy_(new_param)

    # Initialize PyToch class head
    torch_model = OwlViTBoxPredictor(out_dim=4, width=torch_config["vision_width"])
    torch_box_params = torch_model.state_dict()

    # Convert flax params to pytorch
    new_box_head_params = {}
    flax_box_params = flatten_nested_dict(flax_params["obj_box_head"])

    for flax_key, v in flax_box_params.items():
        torch_key = flax_key.replace("/", ".")
        torch_key = torch_key.replace(".kernel", ".weight")
        torch_key = torch_key.replace("_", "").lower()

        if "weight" in torch_key and v.ndim == 2:
            v = v.T

        new_box_head_params[torch_key] = v

    # Copy flax box head params to PyTorch params
    for name, param in new_box_head_params.items():
        if name in torch_box_params.keys():
            new_param = torch.from_numpy(new_box_head_params[name])
            torch_box_params[name].copy_(new_param)

    return torch_class_params, torch_box_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--owlvit_checkpoint", default=None, type=str, required=True, help="Name of flax model."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default="./", type=str, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()

    # Load flax model and print parameters 
    model_name = args.owlvit_checkpoint
    if model_name == "clip_b16":
        config = clip_b16.get_config()
    elif model_name == "clip_b32":
        config = clip_b32.get_config()
    elif model_name == "clip_l14":
        config = clip_l14.get_config()
    else:
        raise Exception("Model not supported")

    # Initialize PyToch clip model
    if model_name == "clip_b16":
        torch_config = CONFIGS["vit_b16"]
    elif model_name == "clip_b32":
        torch_config = CONFIGS["vit_b32"]
    elif model_name == "clip_l14":
        torch_config = CONFIGS["vit_l14"]

    flax_model = models.TextZeroShotDetectionModule(
        body_configs=config.model.body,
        normalize=config.model.normalize,
        box_bias=config.model.box_bias)

    # Load from checkpoint and convert params to float-32
    #variables = flax_model.load_variables(config.init_from.checkpoint_path)
    variables = flax_model.load_variables("checkpoints/clip_vit_b32")
    flax_params = jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, variables['params'])
    del variables
 
    #with torch.no_grad():
    #    img_feats = torch_model.encode_image(torch.zeros(1,3,768,768))
    torch_backbone_params, clip = convert_clip_backbone(flax_params, torch_config)
    torch_class_token_params = convert_embedder(clip, flax_params, config, torch_config)
    torch_class_params, torch_box_params = convert_class_box_heads(flax_params, torch_config)
