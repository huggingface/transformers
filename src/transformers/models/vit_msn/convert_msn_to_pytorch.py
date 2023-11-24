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
"""Convert ViT MSN checkpoints from the original repository: https://github.com/facebookresearch/msn"""

import argparse
import json

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import ViTImageProcessor, ViTMSNConfig, ViTMSNModel
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


torch.set_grad_enabled(False)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"module.blocks.{i}.norm1.weight", f"vit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"module.blocks.{i}.norm1.bias", f"vit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"module.blocks.{i}.attn.proj.weight", f"vit.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append((f"module.blocks.{i}.attn.proj.bias", f"vit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"module.blocks.{i}.norm2.weight", f"vit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"module.blocks.{i}.norm2.bias", f"vit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc1.weight", f"vit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc1.bias", f"vit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc2.weight", f"vit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc2.bias", f"vit.encoder.layer.{i}.output.dense.bias"))

    # projection layer + position embeddings
    rename_keys.extend(
        [
            ("module.cls_token", "vit.embeddings.cls_token"),
            ("module.patch_embed.proj.weight", "vit.embeddings.patch_embeddings.projection.weight"),
            ("module.patch_embed.proj.bias", "vit.embeddings.patch_embeddings.projection.bias"),
            ("module.pos_embed", "vit.embeddings.position_embeddings"),
        ]
    )

    if base_model:
        # layernorm + pooler
        rename_keys.extend(
            [
                ("module.norm.weight", "layernorm.weight"),
                ("module.norm.bias", "layernorm.bias"),
            ]
        )

        # if just the base model, we should remove "vit" from all keys that start with "vit"
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("vit") else pair for pair in rename_keys]
    else:
        # layernorm + classification head
        rename_keys.extend(
            [
                ("norm.weight", "vit.layernorm.weight"),
                ("norm.bias", "vit.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "vit."
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"module.blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"module.blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def remove_projection_head(state_dict):
    # projection head is used in the self-supervised pre-training in MSN,
    # for downstream task it's not needed.
    ignore_keys = [
        "module.fc.fc1.weight",
        "module.fc.fc1.bias",
        "module.fc.bn1.weight",
        "module.fc.bn1.bias",
        "module.fc.bn1.running_mean",
        "module.fc.bn1.running_var",
        "module.fc.bn1.num_batches_tracked",
        "module.fc.fc2.weight",
        "module.fc.fc2.bias",
        "module.fc.bn2.weight",
        "module.fc.bn2.bias",
        "module.fc.bn2.running_mean",
        "module.fc.bn2.running_var",
        "module.fc.bn2.num_batches_tracked",
        "module.fc.fc3.weight",
        "module.fc.fc3.bias",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def convert_vit_msn_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    config = ViTMSNConfig()
    config.num_labels = 1000

    repo_id = "datasets/huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    if "s16" in checkpoint_url:
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_attention_heads = 6
    elif "l16" in checkpoint_url:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.hidden_dropout_prob = 0.1
    elif "b4" in checkpoint_url:
        config.patch_size = 4
    elif "l7" in checkpoint_url:
        config.patch_size = 7
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.hidden_dropout_prob = 0.1

    model = ViTMSNModel(config)

    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["target_encoder"]

    image_processor = ViTImageProcessor(size=config.image_size)

    remove_projection_head(state_dict)
    rename_keys = create_rename_keys(config, base_model=True)

    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model=True)

    model.load_state_dict(state_dict)
    model.eval()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    image = Image.open(requests.get(url, stream=True).raw)
    image_processor = ViTImageProcessor(
        size=config.image_size, image_mean=IMAGENET_DEFAULT_MEAN, image_std=IMAGENET_DEFAULT_STD
    )
    inputs = image_processor(images=image, return_tensors="pt")

    # forward pass
    torch.manual_seed(2)
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state

    # The following Colab Notebook was used to generate these outputs:
    # https://colab.research.google.com/gist/sayakpaul/3672419a04f5997827503fd84079bdd1/scratchpad.ipynb
    if "s16" in checkpoint_url:
        expected_slice = torch.tensor([[-1.0915, -1.4876, -1.1809]])
    elif "b16" in checkpoint_url:
        expected_slice = torch.tensor([[14.2889, -18.9045, 11.7281]])
    elif "l16" in checkpoint_url:
        expected_slice = torch.tensor([[41.5028, -22.8681, 45.6475]])
    elif "b4" in checkpoint_url:
        expected_slice = torch.tensor([[-4.3868, 5.2932, -0.4137]])
    else:
        expected_slice = torch.tensor([[-0.1792, -0.6465, 2.4263]])

    # verify logits
    assert torch.allclose(last_hidden_state[:, 0, :3], expected_slice, atol=1e-4)

    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar",
        type=str,
        help="URL of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_vit_msn_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
