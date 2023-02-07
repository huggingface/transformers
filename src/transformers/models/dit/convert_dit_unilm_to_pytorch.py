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
"""Convert DiT checkpoints from the unilm repository."""


import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import BeitConfig, BeitFeatureExtractor, BeitForImageClassification, BeitForMaskedImageModeling
from transformers.image_utils import PILImageResampling
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, has_lm_head=False, is_semantic=False):
    prefix = "backbone." if is_semantic else ""

    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"{prefix}blocks.{i}.norm1.weight", f"beit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.norm1.bias", f"beit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.weight", f"beit.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.bias", f"beit.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append((f"{prefix}blocks.{i}.norm2.weight", f"beit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.norm2.bias", f"beit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc1.weight", f"beit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc1.bias", f"beit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.weight", f"beit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.bias", f"beit.encoder.layer.{i}.output.dense.bias"))

    # projection layer + position embeddings
    rename_keys.extend(
        [
            (f"{prefix}cls_token", "beit.embeddings.cls_token"),
            (f"{prefix}patch_embed.proj.weight", "beit.embeddings.patch_embeddings.projection.weight"),
            (f"{prefix}patch_embed.proj.bias", "beit.embeddings.patch_embeddings.projection.bias"),
            (f"{prefix}pos_embed", "beit.embeddings.position_embeddings"),
        ]
    )

    if has_lm_head:
        # mask token + layernorm
        rename_keys.extend(
            [
                ("mask_token", "beit.embeddings.mask_token"),
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
            ]
        )
    else:
        # layernorm + classification head
        rename_keys.extend(
            [
                ("fc_norm.weight", "beit.pooler.layernorm.weight"),
                ("fc_norm.bias", "beit.pooler.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, has_lm_head=False, is_semantic=False):
    for i in range(config.num_hidden_layers):
        prefix = "backbone." if is_semantic else ""
        # queries, keys and values
        in_proj_weight = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.v_bias")

        state_dict[f"beit.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"beit.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"beit.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"beit.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"beit.encoder.layer.{i}.attention.attention.value.bias"] = v_bias

        # gamma_1 and gamma_2
        # we call them lambda because otherwise they are renamed when using .from_pretrained
        gamma_1 = state_dict.pop(f"{prefix}blocks.{i}.gamma_1")
        gamma_2 = state_dict.pop(f"{prefix}blocks.{i}.gamma_2")

        state_dict[f"beit.encoder.layer.{i}.lambda_1"] = gamma_1
        state_dict[f"beit.encoder.layer.{i}.lambda_2"] = gamma_2


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_dit_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our BEiT structure.
    """

    # define default BEiT configuration
    has_lm_head = False if "rvlcdip" in checkpoint_url else True
    config = BeitConfig(use_absolute_position_embeddings=True, use_mask_token=has_lm_head)

    # size of the architecture
    if "large" in checkpoint_url or "dit-l" in checkpoint_url:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16

    # labels
    if "rvlcdip" in checkpoint_url:
        config.num_labels = 16
        repo_id = "huggingface/label-files"
        filename = "rvlcdip-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    # load state_dict of original model, remove and rename some keys
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]

    rename_keys = create_rename_keys(config, has_lm_head=has_lm_head)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, has_lm_head=has_lm_head)

    # load HuggingFace model
    model = BeitForMaskedImageModeling(config) if has_lm_head else BeitForImageClassification(config)
    model.eval()
    model.load_state_dict(state_dict)

    # Check outputs on an image
    feature_extractor = BeitFeatureExtractor(
        size=config.image_size, resample=PILImageResampling.BILINEAR, do_center_crop=False
    )
    image = prepare_img()

    encoding = feature_extractor(images=image, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    outputs = model(pixel_values)
    logits = outputs.logits

    # verify logits
    expected_shape = [1, 16] if "rvlcdip" in checkpoint_url else [1, 196, 8192]
    assert logits.shape == torch.Size(expected_shape), "Shape of logits not as expected"

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        if has_lm_head:
            model_name = "dit-base" if "base" in checkpoint_url else "dit-large"
        else:
            model_name = "dit-base-finetuned-rvlcdip" if "dit-b" in checkpoint_url else "dit-large-finetuned-rvlcdip"
        feature_extractor.push_to_hub(
            repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
            organization="nielsr",
            commit_message="Add feature extractor",
            use_temp_dir=True,
        )
        model.push_to_hub(
            repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
            organization="nielsr",
            commit_message="Add model",
            use_temp_dir=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    args = parser.parse_args()
    convert_dit_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
