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
"""Convert DPT checkpoints from the original repository. URL: https://github.com/isl-org/DPT"""


import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import cached_download, hf_hub_url
from transformers import DPTConfig, DPTForDepthEstimation
from transformers.utils import logging

from torchvision.transforms import Compose, Resize, ToTensor, Normalize


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config(checkpoint_url):
    config = DPTConfig()

    if "large" in checkpoint_url:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.out_indices = [5, 11, 17, 23]
        config.post_process_channels = [256, 512, 1024, 1024]

    # TODO set id2label and label2id

    return config


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "pretrained.model.norm.weight",
        "pretrained.model.norm.bias",
        "pretrained.model.head.weight",
        "pretrained.model.head.bias",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(name):
    if "pretrained.model" in name and "cls_token" not in name and "pos_embed" not in name and "patch_embed" not in name:
        name = name.replace("pretrained.model", "dpt.encoder")
    if "pretrained.model" in name:
        name = name.replace("pretrained.model", "dpt.embeddings")
    if "patch_embed" in name:
        name = name.replace("patch_embed", "patch_embeddings")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "position_embeddings")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "proj" in name and "project" not in name:
        name = name.replace("proj", "projection")
    if "blocks" in name:
        name = name.replace("blocks", "layer")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "scratch.output_conv" in name:
        name = name.replace("scratch.output_conv", "head")
    if "scratch" in name:
        name = name.replace("scratch", "dpt")
    if "layer1_rn" in name:
        name = name.replace("layer1_rn", "convs.0")
    if "layer2_rn" in name:
        name = name.replace("layer2_rn", "convs.1")
    if "layer3_rn" in name:
        name = name.replace("layer3_rn", "convs.2")
    if "layer4_rn" in name:
        name = name.replace("layer4_rn", "convs.3")
    if "refinenet" in name:
        layer_idx = int(name[len("dpt.refinenet"):len("dpt.refinenet") + 1])
        # tricky here: we need to map 4 to 0, 3 to 1, 2 to 2 and 1 to 3
        name = name.replace(f"refinenet{layer_idx}", f"fusion_blocks.{abs(layer_idx-4)}")
    if "out_conv" in name:
        name = name.replace("out_conv", "project")
    if "resConfUnit1" in name:
        name = name.replace("resConfUnit1", "res_conv_unit1")
    if "resConfUnit2" in name:
        name = name.replace("resConfUnit2", "res_conv_unit2")
    if "pretrained" in name:
        name = name.replace("pretrained", "dpt")
    # readout blocks
    if "act_postprocess1.0.project.0" in name:
        name = name.replace("act_postprocess1.0.project.0", "reassemble_blocks.readout_projects.0.0")
    if "act_postprocess2.0.project.0" in name:
        name = name.replace("act_postprocess2.0.project.0", "reassemble_blocks.readout_projects.1.0")
    if "act_postprocess3.0.project.0" in name:
        name = name.replace("act_postprocess3.0.project.0", "reassemble_blocks.readout_projects.2.0")
    if "act_postprocess4.0.project.0" in name:
        name = name.replace("act_postprocess4.0.project.0", "reassemble_blocks.readout_projects.3.0")
    # resize blocks
    if "act_postprocess1.3" in name:
        name = name.replace("act_postprocess1.3", "reassemble_blocks.projects.0")
    if "act_postprocess1.4" in name:
        name = name.replace("act_postprocess1.4", "reassemble_blocks.resize_layers.0")
    if "act_postprocess2.3" in name:
        name = name.replace("act_postprocess2.3", "reassemble_blocks.projects.1")
    if "act_postprocess2.4" in name:
        name = name.replace("act_postprocess2.4", "reassemble_blocks.resize_layers.1")
    if "act_postprocess3.3" in name:
        name = name.replace("act_postprocess3.3", "reassemble_blocks.projects.2")
    if "act_postprocess4.3" in name:
        name = name.replace("act_postprocess4.3", "reassemble_blocks.projects.3")
    if "act_postprocess4.4" in name:
        name = name.replace("act_postprocess4.4", "reassemble_blocks.resize_layers.3")

    return name


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    for i in range(config.num_hidden_layers):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_dpt_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    # define DPT configuration based on URL
    config = get_dpt_config(checkpoint_url)
    # load original state_dict from URL
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # remove certain keys
    remove_ignore_keys_(state_dict)
    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # read in qkv matrices
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    model = DPTForDepthEstimation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # Check outputs on an image
    # TODO prepare image by DPTFeatureExtractor
    image = prepare_img()

    transform = Compose([
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    pixel_values = transform(image).unsqueeze(0)

    # forward pass
    logits = model(pixel_values).logits

    print("Shape of logits:", logits.shape)
    print("First elements of logits:", logits[0,:3,:3])

    # TODO assert logits
    expected_slice = torch.tensor([[6.3199, 6.3629, 6.4148],
        [6.3850, 6.3615, 6.4166],
        [6.3519, 6.3176, 6.3575]])
    assert torch.allclose(logits[0,:3,:3], expected_slice)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    #print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    #feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model to hub...")
        model_name = "dpt-large-ade"
        model.push_to_hub(
            repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
            organization="nielsr",
            commit_message="Add model",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        type=str,
        help="URL of the original DPT checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        default=False,
        type=bool,
        required=False,
        help="Whether to push the model to the hub.",
    )


    args = parser.parse_args()
    convert_dpt_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)