# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert FastViT from the timm library."""

import argparse
import json
from pathlib import Path

import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import FastViTConfig, FastViTForImageClassification, ViTFeatureExtractor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_fastvit_config(fastvit_name):
    # define default FastViT configuration
    config = FastViTConfig()
    config.num_labels = 1000
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    config.image_size = 256

    # size of the architecture
    if "t8" in fastvit_name:
        pass
    elif "t12" in fastvit_name:
        config.depths = [2, 2, 6, 2]
        config.hidden_sizes = [64, 128, 256, 512]
    elif "s12" in fastvit_name:
        config.depths = [2, 2, 6, 2]
        config.hidden_sizes = [64, 128, 256, 512]
        config.mlp_ratio = 4.0
    elif "sa12" in fastvit_name:
        config.depths = [2, 2, 6, 2]
        config.hidden_sizes = [64, 128, 256, 512]
        config.mlp_ratio = 4.0
        config.pos_embeds = [None, None, None, "RepCPE"]
        config.token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    elif "sa24" in fastvit_name:
        config.depths = [4, 4, 12, 4]
        config.hidden_sizes = [64, 128, 256, 512]
        config.mlp_ratio = 4.0
        config.pos_embeds = [None, None, None, "RepCPE"]
        config.token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    elif "sa36" in fastvit_name:
        config.depths = [6, 6, 18, 6]
        config.hidden_sizes = [64, 128, 256, 512]
        config.mlp_ratio = 4.0
        config.pos_embeds = [None, None, None, "RepCPE"]
        config.token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    elif "ma36" in fastvit_name:
        config.depths = [6, 6, 18, 6]
        config.hidden_sizes = [76, 152, 304, 608]
        config.mlp_ratio = 4.0
        config.pos_embeds = [None, None, None, "RepCPE"]
        config.token_mixers = ("repmixer", "repmixer", "repmixer", "attention")

    return config


def rename_key(name):
    if "stem" in name:
        name = name.replace("stem", "embeddings.patch_embeddings.projection")
    if "conv_kxk" in name:
        name = name.replace("conv_kxk", "rbr_conv")
    if "conv_scale" in name:
        name = name.replace("conv_scale", "rbr_scale")
    if "identity" in name:
        name = name.replace("identity", "rbr_skip")
    if "0.conv" in name:
        name = name.replace("0.conv", "conv")
    if "0.bn" in name:
        name = name.replace("0.bn", "bn")
    if "stages" in name:
        name = name.replace("stages", "encoder.layer")
    if "blocks" in name:
        name = name.replace("blocks", "stage_conv")
    if "layer_scale.gamma" in name:
        name = name.replace("layer_scale.gamma", "layer_scale")
        name = name.replace("token_mixer", "token_mixer_block.token_mixer")
    if "layer_scale_1.gamma" in name:
        name = name.replace("layer_scale_1.gamma", "layer_scale_1")
    if "layer_scale_2.gamma" in name:
        name = name.replace("layer_scale_2.gamma", "layer_scale_2")
    if "token_mixer.norm" in name:
        name = name.replace("token_mixer.norm", "token_mixer_block.token_mixer.norm")
    if "token_mixer.mixer" in name:
        name = name.replace("token_mixer.mixer", "token_mixer_block.token_mixer.mixer")
    if "mlp" in name:
        name = name.replace("mlp", "convffn")
    if ".conv.conv" in name:
        name = name.replace("conv.conv", "conv")
    if ".conv.bn" in name:
        name = name.replace("conv.bn", "bn")
    if "proj." in name:
        if "token_mixer" not in name:
            name_split = name.split(".")
            pos = int(name_split[2])
            name_split[2] = str(pos - 1)
            if int(name_split[5]) == 0:
                name_split[4] = "reparam_large_conv"
            else:
                name_split[4] = "conv"
            name_split.pop(5)  # drop the 0 or 1....
            name = ".".join(name_split)
        else:
            name = name.replace("token_mixer.proj", "token_mixer_block.attention.proj")
    if "se.fc1" in name:
        name = name.replace("se.fc1", "se.reduce")
    if "se.fc2" in name:
        name = name.replace("se.fc2", "se.expand")
    if "q_bias" in name:
        name = name.replace("q_bias", "query.bias")
    if "k_bias" in name:
        name = name.replace("k_bias", "key.bias")
    if "v_bias" in name:
        name = name.replace("v_bias", "value.bias")
    if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
        name_split = name.split(".")
        if len(name_split) > 5:
            if name_split[5].isdigit():
                name_split.pop(5)  # drop the 0 or 1....
        name = ".".join(name_split)
    if "norm" in name and "token_mixer" not in name:
        name_split = name.split(".")
        if name_split[4].isdigit():
            name = name.replace("norm", "token_mixer_block.norm")
    if "head.fc" in name or "final_conv" in name:
        name = name.replace("head.fc", "classifier")
    else:
        name = "fastvit." + name
    return name


def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)
        if "mask" in key:
            continue
        elif "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[1])
            block_num = int(key_split[3])

            if "weight" in key:
                orig_state_dict[
                    f"fastvit.encoder.layer.{layer_num}.stage_conv.{block_num}.token_mixer_block.attention.qkv.weight"
                ] = val
            else:
                orig_state_dict[
                    f"fastvit.encoder.layer.{layer_num}.stage_conv.{block_num}.token_mixer_block.attention.qkv.bias"
                ] = val
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_fastvit_checkpoint(fastvit_name, pytorch_dump_folder_path):
    # load original model from timm
    timm_model = timm.create_model(fastvit_name, pretrained=True)
    timm_model.eval()

    # load model from HF
    config = get_fastvit_config(fastvit_name)
    model = FastViTForImageClassification(config)
    model.eval()

    # load state_dict of original model, remove and rename some keys
    state_dict = timm_model.state_dict()
    new_state_dict = convert_state_dict(state_dict, model)

    model.load_state_dict(new_state_dict)

    feature_extractor = ViTFeatureExtractor(size=config.image_size)
    inputs = feature_extractor(images=prepare_img(), return_tensors="pt")

    outputs = model(**inputs).logits
    timm_logits = timm_model(inputs["pixel_values"])

    assert outputs.shape == timm_logits.shape, f"Shape is not equal: {outputs.shape} and {timm_logits.shape}"
    assert torch.allclose(timm_logits, outputs, atol=1e-3), "The predicted logits are not the same."

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {fastvit_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fastvit_name",
        default="timm/fastvit_t8.apple_in1k",
        type=str,
        help="Name of the FastViT timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_fastvit_checkpoint(args.fastvit_name, args.pytorch_dump_folder_path)
