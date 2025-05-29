# coding=utf-8
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
"""Convert Hiera checkpoints from the original repository.

URL: https://github.com/facebookresearch/hiera
"""

import argparse
import json
import math
from typing import Dict, Tuple

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import BitImageProcessor, HieraConfig, HieraForImageClassification, HieraForPreTraining, HieraModel
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config: HieraConfig, base_model: bool, mae_model: bool):
    rename_keys = []
    # fmt: off
    num_stages = len(config.depths)
    # embedding dimensions for input and stages
    dims = [config.embed_dim] + [int(config.embed_dim * config.embed_dim_multiplier**i) for i in range(num_stages)]

    global_layer_idx = 0
    for stage_idx in range(num_stages):
        dim_in = dims[stage_idx]
        dim_out = dims[stage_idx + 1]
        for layer_idx in range(config.depths[stage_idx]):
            rename_keys.append((f"blocks.{global_layer_idx}.norm1.weight", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.layernorm_before.weight"))
            rename_keys.append((f"blocks.{global_layer_idx}.norm1.bias", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.layernorm_before.bias"))
            rename_keys.append((f"blocks.{global_layer_idx}.attn.qkv.weight", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.attn.qkv.weight"))
            rename_keys.append((f"blocks.{global_layer_idx}.attn.qkv.bias", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.attn.qkv.bias"))
            rename_keys.append((f"blocks.{global_layer_idx}.attn.proj.weight", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.attn.proj.weight"))
            rename_keys.append((f"blocks.{global_layer_idx}.attn.proj.bias", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.attn.proj.bias"))
            rename_keys.append((f"blocks.{global_layer_idx}.norm2.weight", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.layernorm_after.weight"))
            rename_keys.append((f"blocks.{global_layer_idx}.norm2.bias", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.layernorm_after.bias"))
            rename_keys.append((f"blocks.{global_layer_idx}.mlp.fc1.weight", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.mlp.fc1.weight"))
            rename_keys.append((f"blocks.{global_layer_idx}.mlp.fc1.bias", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.mlp.fc1.bias"))
            rename_keys.append((f"blocks.{global_layer_idx}.mlp.fc2.weight", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.mlp.fc2.weight"))
            rename_keys.append((f"blocks.{global_layer_idx}.mlp.fc2.bias", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.mlp.fc2.bias"))

            # projection layer only for the first layer of each stage boundary (except the first stage)
            if dim_out != dim_in and layer_idx == 0:
                rename_keys.append((f"blocks.{global_layer_idx}.proj.weight", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.proj.weight"))
                rename_keys.append((f"blocks.{global_layer_idx}.proj.bias", f"hiera.encoder.stages.{stage_idx}.layers.{layer_idx}.proj.bias"))

            global_layer_idx += 1

    # projection layer + position embeddings
    rename_keys.extend(
        [
            ("patch_embed.proj.weight", "hiera.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "hiera.embeddings.patch_embeddings.projection.bias")
        ]
    )

    rename_keys.append(("pos_embed", "hiera.embeddings.position_embeddings"))

    if base_model:
        # layernorm + pooler
        rename_keys.extend([("norm.weight", "pooler.layernorm.weight"), ("norm.bias", "pooler.layernorm.bias")])
        # if just the base model, we should remove "hiera" from all keys that start with "hiera"
        rename_keys = [(pair[0], pair[1][6:]) if pair[1].startswith("hiera") else pair for pair in rename_keys]
    elif mae_model:
        rename_keys.extend(
            [
                ("encoder_norm.weight", "encoder_norm.weight"),
                ("encoder_norm.bias", "encoder_norm.bias"),
                ("mask_token", "decoder.mask_token"),
                ("decoder_pos_embed", "decoder.decoder_position_embeddings"),
                ("decoder_norm.weight", "decoder.decoder_norm.weight"),
                ("decoder_norm.bias", "decoder.decoder_norm.bias"),
                ("decoder_pred.weight", "decoder.decoder_pred.weight"),
                ("decoder_pred.bias", "decoder.decoder_pred.bias"),
                ("decoder_embed.weight", "decoder.decoder_embeddings.weight"),
                ("decoder_embed.bias", "decoder.decoder_embeddings.bias")
            ]
        )
        for i in range(config.decoder_depth):
            rename_keys.extend(
                [
                    (f"decoder_blocks.{i}.norm1.weight", f"decoder.decoder_block.layers.{i}.layernorm_before.weight"),
                    (f"decoder_blocks.{i}.norm1.bias", f"decoder.decoder_block.layers.{i}.layernorm_before.bias"),
                    (f"decoder_blocks.{i}.attn.qkv.weight", f"decoder.decoder_block.layers.{i}.attn.qkv.weight"),
                    (f"decoder_blocks.{i}.attn.qkv.bias", f"decoder.decoder_block.layers.{i}.attn.qkv.bias"),
                    (f"decoder_blocks.{i}.attn.proj.weight", f"decoder.decoder_block.layers.{i}.attn.proj.weight"),
                    (f"decoder_blocks.{i}.attn.proj.bias", f"decoder.decoder_block.layers.{i}.attn.proj.bias"),
                    (f"decoder_blocks.{i}.norm2.weight", f"decoder.decoder_block.layers.{i}.layernorm_after.weight"),
                    (f"decoder_blocks.{i}.norm2.bias", f"decoder.decoder_block.layers.{i}.layernorm_after.bias"),
                    (f"decoder_blocks.{i}.mlp.fc1.weight", f"decoder.decoder_block.layers.{i}.mlp.fc1.weight"),
                    (f"decoder_blocks.{i}.mlp.fc1.bias", f"decoder.decoder_block.layers.{i}.mlp.fc1.bias"),
                    (f"decoder_blocks.{i}.mlp.fc2.weight", f"decoder.decoder_block.layers.{i}.mlp.fc2.weight"),
                    (f"decoder_blocks.{i}.mlp.fc2.bias", f"decoder.decoder_block.layers.{i}.mlp.fc2.bias"),
                ]
            )
        for i in range(config.num_query_pool):
            rename_keys.extend(
                [
                    (f"multi_scale_fusion_heads.{i}.weight", f"multiscale_fusion.multi_scale_fusion_heads.{i}.weight"),
                    (f"multi_scale_fusion_heads.{i}.bias", f"multiscale_fusion.multi_scale_fusion_heads.{i}.bias")
                ]
            )
    else:
        # layernorm + classification head
        rename_keys.extend(
            [
                ("norm.weight", "hiera.pooler.layernorm.weight"),
                ("norm.bias", "hiera.pooler.layernorm.bias"),
                ("head.projection.weight", "classifier.weight"),
                ("head.projection.bias", "classifier.bias"),
            ]
        )
    # fmt: on
    return rename_keys


def remove_classification_head_(state_dict):
    ignore_keys = ["head.projection.weight", "head.projection.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


def get_labels_for_classifier(model_name: str) -> Tuple[Dict[int, str], Dict[str, int], int]:
    repo_id = "huggingface/label-files"

    filename = "imagenet-1k-id2label.json"

    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    return id2label, label2id, num_labels


def get_hiera_config(model_name: str, base_model: bool, mae_model: bool) -> HieraConfig:
    if model_name == "hiera-tiny-224":
        config = HieraConfig(depths=[1, 2, 7, 2])
    elif model_name == "hiera-small-224":
        config = HieraConfig(depths=[1, 2, 11, 2])
    elif model_name == "hiera-base-224":
        config = HieraConfig()
    elif model_name == "hiera-base-plus-224":
        config = HieraConfig(embed_dim=112, num_heads=[2, 4, 8, 16])
    elif model_name == "hiera-large-224":
        config = HieraConfig(embed_dim=144, num_heads=[2, 4, 8, 16], depths=[2, 6, 36, 4])
    elif model_name == "hiera-huge-224":
        config = HieraConfig(embed_dim=256, num_heads=[4, 8, 16, 32], depths=[2, 6, 36, 4])
    else:
        raise ValueError(f"Unrecognized model name: {model_name}")

    if base_model:
        pass
    elif mae_model:
        config.num_query_pool = 2
        config.decoder_hidden_size = 512
        config.decoder_depth = 8
        config.decoder_num_heads = 16
        # Table 3b from Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
        config.mask_ratio = 0.6
    else:
        id2label, label2id, num_labels = get_labels_for_classifier(model_name)
        config.id2label = id2label
        config.label2id = label2id
        config.num_labels = num_labels

    return config


@torch.no_grad()
def convert_hiera_checkpoint(args):
    model_name = args.model_name
    base_model = args.base_model
    pytorch_dump_folder_path = args.pytorch_dump_folder_path
    push_to_hub = args.push_to_hub
    mae_model = args.mae_model

    config = get_hiera_config(model_name, base_model, mae_model)

    # Load original hiera model
    original_model_name = model_name.replace("-", "_")
    original_model_name = f"mae_{original_model_name}" if mae_model else original_model_name

    original_checkpoint_name = "mae_in1k_ft_in1k" if not (base_model or mae_model) else "mae_in1k"

    original_model = torch.hub.load(
        "facebookresearch/hiera",
        model=original_model_name,
        pretrained=True,
        checkpoint=original_checkpoint_name,
    )

    original_model.eval()
    original_state_dict = original_model.state_dict()
    # Don't need to remove head for MAE because original implementation doesn't have it on MAE
    if base_model:
        remove_classification_head_(original_state_dict)

    # # Rename keys
    new_state_dict = original_state_dict.copy()
    rename_keys = create_rename_keys(config, base_model, mae_model)

    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)

    # Load HF hiera model
    if base_model:
        model = HieraModel(config)
    elif mae_model:
        model = HieraForPreTraining(config)
    else:
        model = HieraForImageClassification(config)

    model.eval()

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    input_image = prepare_img()

    original_image_preprocessor = transforms.Compose(
        [
            transforms.Resize(int((256 / 224) * 224), interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )

    image_processor = BitImageProcessor(
        image_mean=IMAGENET_DEFAULT_MEAN, image_std=IMAGENET_DEFAULT_STD, size={"shortest_edge": 256}
    )
    inputs = image_processor(images=input_image, return_tensors="pt")

    expected_pixel_values = original_image_preprocessor(input_image).unsqueeze(0)

    input_image = prepare_img()

    inputs = image_processor(images=input_image, return_tensors="pt")
    expected_pixel_values = original_image_preprocessor(input_image).unsqueeze(0)
    assert torch.allclose(inputs.pixel_values, expected_pixel_values, atol=1e-4)
    print("Pixel values look good!")
    print(f"{inputs.pixel_values[0, :3, :3, :3]=}")

    # If is MAE we pass a noise to generate a random mask
    mask_spatial_shape = [
        i // s // ms for i, s, ms in zip(config.image_size, config.patch_stride, config.masked_unit_size)
    ]
    num_windows = math.prod(mask_spatial_shape)
    torch.manual_seed(2)
    noise = torch.rand(1, num_windows)
    outputs = model(**inputs) if not mae_model else model(noise=noise, **inputs)
    # original implementation returns logits.softmax(dim=-1)

    if base_model:
        expected_prob, expected_intermediates = original_model(expected_pixel_values, return_intermediates=True)
        expected_last_hidden = expected_intermediates[-1]
        batch_size, _, _, hidden_dim = expected_last_hidden.shape
        expected_last_hidden = expected_last_hidden.reshape(batch_size, -1, hidden_dim)
        assert torch.allclose(outputs.last_hidden_state, expected_last_hidden, atol=1e-3)
        print("Base Model looks good as hidden states match original implementation!")
        print(f"{outputs.last_hidden_state[0, :3, :3]=}")
    elif mae_model:
        # get mask from noise to be able to compare outputs
        mask, _ = model.hiera.embeddings.patch_embeddings.random_masking(expected_pixel_values, noise)
        expected_loss, _, _, _ = original_model(expected_pixel_values, mask=mask.bool())
        assert torch.allclose(outputs.loss, expected_loss, atol=1e-3)
        print("MAE Model looks good as loss matches original implementation!")
    else:
        expected_prob = original_model(expected_pixel_values)
        assert torch.allclose(outputs.logits.softmax(dim=-1), expected_prob, atol=1e-3)
        print("Classifier looks good as probs match original implementation")
        print(f"{outputs.logits[:, :5]=}")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor for {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        hub_name = model_name
        if base_model:
            hub_name = model_name
        elif mae_model:
            hub_name = f"{model_name}-mae"
        else:
            hub_name = f"{model_name}-in1k"
        repo_id = f"EduardoPacheco/{hub_name}"
        print(f"Pushing model and processor for {model_name} to hub at {repo_id}")
        model.push_to_hub(repo_id)
        image_processor.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model-name",
        default="hiera-tiny-224",
        type=str,
        choices=[
            "hiera-tiny-224",
            "hiera-small-224",
            "hiera-base-224",
            "hiera-base-plus-224",
            "hiera-large-224",
            "hiera-huge-224",
        ],
        help="Name of the Hiera model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch-dump-folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--verify-logits",
        action="store_true",
        help="Whether or not to verify the logits against the original implementation.",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Whether to only convert the base model (no projection head weights).",
    )
    parser.add_argument(
        "--mae-model", action="store_true", help="Whether to convert to MAE checkpoint to HieraForPreTraining."
    )

    args = parser.parse_args()
    convert_hiera_checkpoint(args)
