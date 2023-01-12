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
"""Convert MobileViT checkpoints from the ml-cvnets library."""


import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import (
    MobileViTConfig,
    MobileViTFeatureExtractor,
    MobileViTForImageClassification,
    MobileViTForSemanticSegmentation,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_mobilevit_config(mobilevit_name):
    config = MobileViTConfig()

    # size of the architecture
    if "mobilevit_s" in mobilevit_name:
        config.hidden_sizes = [144, 192, 240]
        config.neck_hidden_sizes = [16, 32, 64, 96, 128, 160, 640]
    elif "mobilevit_xs" in mobilevit_name:
        config.hidden_sizes = [96, 120, 144]
        config.neck_hidden_sizes = [16, 32, 48, 64, 80, 96, 384]
    elif "mobilevit_xxs" in mobilevit_name:
        config.hidden_sizes = [64, 80, 96]
        config.neck_hidden_sizes = [16, 16, 24, 48, 64, 80, 320]
        config.hidden_dropout_prob = 0.05
        config.expand_ratio = 2.0

    if mobilevit_name.startswith("deeplabv3_"):
        config.image_size = 512
        config.output_stride = 16
        config.num_labels = 21
        filename = "pascal-voc-id2label.json"
    else:
        config.num_labels = 1000
        filename = "imagenet-1k-id2label.json"

    repo_id = "huggingface/label-files"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


def rename_key(name, base_model=False):
    for i in range(1, 6):
        if f"layer_{i}." in name:
            name = name.replace(f"layer_{i}.", f"encoder.layer.{i - 1}.")

    if "conv_1." in name:
        name = name.replace("conv_1.", "conv_stem.")
    if ".block." in name:
        name = name.replace(".block.", ".")

    if "exp_1x1" in name:
        name = name.replace("exp_1x1", "expand_1x1")
    if "red_1x1" in name:
        name = name.replace("red_1x1", "reduce_1x1")
    if ".local_rep.conv_3x3." in name:
        name = name.replace(".local_rep.conv_3x3.", ".conv_kxk.")
    if ".local_rep.conv_1x1." in name:
        name = name.replace(".local_rep.conv_1x1.", ".conv_1x1.")
    if ".norm." in name:
        name = name.replace(".norm.", ".normalization.")
    if ".conv." in name:
        name = name.replace(".conv.", ".convolution.")
    if ".conv_proj." in name:
        name = name.replace(".conv_proj.", ".conv_projection.")

    for i in range(0, 2):
        for j in range(0, 4):
            if f".{i}.{j}." in name:
                name = name.replace(f".{i}.{j}.", f".{i}.layer.{j}.")

    for i in range(2, 6):
        for j in range(0, 4):
            if f".{i}.{j}." in name:
                name = name.replace(f".{i}.{j}.", f".{i}.")
                if "expand_1x1" in name:
                    name = name.replace("expand_1x1", "downsampling_layer.expand_1x1")
                if "conv_3x3" in name:
                    name = name.replace("conv_3x3", "downsampling_layer.conv_3x3")
                if "reduce_1x1" in name:
                    name = name.replace("reduce_1x1", "downsampling_layer.reduce_1x1")

    for i in range(2, 5):
        if f".global_rep.{i}.weight" in name:
            name = name.replace(f".global_rep.{i}.weight", ".layernorm.weight")
        if f".global_rep.{i}.bias" in name:
            name = name.replace(f".global_rep.{i}.bias", ".layernorm.bias")

    if ".global_rep." in name:
        name = name.replace(".global_rep.", ".transformer.")
    if ".pre_norm_mha.0." in name:
        name = name.replace(".pre_norm_mha.0.", ".layernorm_before.")
    if ".pre_norm_mha.1.out_proj." in name:
        name = name.replace(".pre_norm_mha.1.out_proj.", ".attention.output.dense.")
    if ".pre_norm_ffn.0." in name:
        name = name.replace(".pre_norm_ffn.0.", ".layernorm_after.")
    if ".pre_norm_ffn.1." in name:
        name = name.replace(".pre_norm_ffn.1.", ".intermediate.dense.")
    if ".pre_norm_ffn.4." in name:
        name = name.replace(".pre_norm_ffn.4.", ".output.dense.")
    if ".transformer." in name:
        name = name.replace(".transformer.", ".transformer.layer.")

    if ".aspp_layer." in name:
        name = name.replace(".aspp_layer.", ".")
    if ".aspp_pool." in name:
        name = name.replace(".aspp_pool.", ".")
    if "seg_head." in name:
        name = name.replace("seg_head.", "segmentation_head.")
    if "segmentation_head.classifier.classifier." in name:
        name = name.replace("segmentation_head.classifier.classifier.", "segmentation_head.classifier.")

    if "classifier.fc." in name:
        name = name.replace("classifier.fc.", "classifier.")
    elif (not base_model) and ("segmentation_head." not in name):
        name = "mobilevit." + name

    return name


def convert_state_dict(orig_state_dict, model, base_model=False):
    if base_model:
        model_prefix = ""
    else:
        model_prefix = "mobilevit."

    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if key[:8] == "encoder.":
            key = key[8:]

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[0][6:]) - 1
            transformer_num = int(key_split[3])
            layer = model.get_submodule(f"{model_prefix}encoder.layer.{layer_num}")
            dim = layer.transformer.layer[transformer_num].attention.attention.all_head_size
            prefix = (
                f"{model_prefix}encoder.layer.{layer_num}.transformer.layer.{transformer_num}.attention.attention."
            )
            if "weight" in key:
                orig_state_dict[prefix + "query.weight"] = val[:dim, :]
                orig_state_dict[prefix + "key.weight"] = val[dim : dim * 2, :]
                orig_state_dict[prefix + "value.weight"] = val[-dim:, :]
            else:
                orig_state_dict[prefix + "query.bias"] = val[:dim]
                orig_state_dict[prefix + "key.bias"] = val[dim : dim * 2]
                orig_state_dict[prefix + "value.bias"] = val[-dim:]
        else:
            orig_state_dict[rename_key(key, base_model)] = val

    return orig_state_dict


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_movilevit_checkpoint(mobilevit_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our MobileViT structure.
    """
    config = get_mobilevit_config(mobilevit_name)

    # load original state_dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # load 🤗 model
    if mobilevit_name.startswith("deeplabv3_"):
        model = MobileViTForSemanticSegmentation(config).eval()
    else:
        model = MobileViTForImageClassification(config).eval()

    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # Check outputs on an image, prepared by MobileViTFeatureExtractor
    feature_extractor = MobileViTFeatureExtractor(crop_size=config.image_size, size=config.image_size + 32)
    encoding = feature_extractor(images=prepare_img(), return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits

    if mobilevit_name.startswith("deeplabv3_"):
        assert logits.shape == (1, 21, 32, 32)

        if mobilevit_name == "deeplabv3_mobilevit_s":
            expected_logits = torch.tensor(
                [
                    [[6.2065, 6.1292, 6.2070], [6.1079, 6.1254, 6.1747], [6.0042, 6.1071, 6.1034]],
                    [[-6.9253, -6.8653, -7.0398], [-7.3218, -7.3983, -7.3670], [-7.1961, -7.2482, -7.1569]],
                    [[-4.4723, -4.4348, -4.3769], [-5.3629, -5.4632, -5.4598], [-5.1587, -5.3402, -5.5059]],
                ]
            )
        elif mobilevit_name == "deeplabv3_mobilevit_xs":
            expected_logits = torch.tensor(
                [
                    [[5.4449, 5.5733, 5.6314], [5.1815, 5.3930, 5.5963], [5.1656, 5.4333, 5.4853]],
                    [[-9.4423, -9.7766, -9.6714], [-9.1581, -9.5720, -9.5519], [-9.1006, -9.6458, -9.5703]],
                    [[-7.7721, -7.3716, -7.1583], [-8.4599, -8.0624, -7.7944], [-8.4172, -7.8366, -7.5025]],
                ]
            )
        elif mobilevit_name == "deeplabv3_mobilevit_xxs":
            expected_logits = torch.tensor(
                [
                    [[6.9811, 6.9743, 7.3123], [7.1777, 7.1931, 7.3938], [7.5633, 7.8050, 7.8901]],
                    [[-10.5536, -10.2332, -10.2924], [-10.2336, -9.8624, -9.5964], [-10.8840, -10.8158, -10.6659]],
                    [[-3.4938, -3.0631, -2.8620], [-3.4205, -2.8135, -2.6875], [-3.4179, -2.7945, -2.8750]],
                ]
            )
        else:
            raise ValueError(f"Unknown mobilevit_name: {mobilevit_name}")

        assert torch.allclose(logits[0, :3, :3, :3], expected_logits, atol=1e-4)
    else:
        assert logits.shape == (1, 1000)

        if mobilevit_name == "mobilevit_s":
            expected_logits = torch.tensor([-0.9866, 0.2392, -1.1241])
        elif mobilevit_name == "mobilevit_xs":
            expected_logits = torch.tensor([-2.4761, -0.9399, -1.9587])
        elif mobilevit_name == "mobilevit_xxs":
            expected_logits = torch.tensor([-1.9364, -1.2327, -0.4653])
        else:
            raise ValueError(f"Unknown mobilevit_name: {mobilevit_name}")

        assert torch.allclose(logits[0, :3], expected_logits, atol=1e-4)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {mobilevit_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model_mapping = {
            "mobilevit_s": "mobilevit-small",
            "mobilevit_xs": "mobilevit-x-small",
            "mobilevit_xxs": "mobilevit-xx-small",
            "deeplabv3_mobilevit_s": "deeplabv3-mobilevit-small",
            "deeplabv3_mobilevit_xs": "deeplabv3-mobilevit-x-small",
            "deeplabv3_mobilevit_xxs": "deeplabv3-mobilevit-xx-small",
        }

        print("Pushing to the hub...")
        model_name = model_mapping[mobilevit_name]
        feature_extractor.push_to_hub(model_name, organization="apple")
        model.push_to_hub(model_name, organization="apple")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--mobilevit_name",
        default="mobilevit_s",
        type=str,
        help=(
            "Name of the MobileViT model you'd like to convert. Should be one of 'mobilevit_s', 'mobilevit_xs',"
            " 'mobilevit_xxs', 'deeplabv3_mobilevit_s', 'deeplabv3_mobilevit_xs', 'deeplabv3_mobilevit_xxs'."
        ),
    )
    parser.add_argument(
        "--checkpoint_path", required=True, type=str, help="Path to the original state dict (.pt file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_movilevit_checkpoint(
        args.mobilevit_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
