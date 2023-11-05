# coding=utf-8
# Copyright 2023 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
import copy
import json
import logging

import requests
import torch
from PIL import Image

from transformers import FastConfig, FastForSceneTextRecognition
from transformers.models.fast.image_processing_fast import FastImageProcessor


tiny_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_tiny.config"
small_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_small.config"
base_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_base.config"

rename_key_mappings = {
    "bn": "batch_norm",
    "hor": "horizontal",
    "ver": "vertical",
}


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


def prepare_config(size_config_url, pooling_size, min_area, bbox_type, loss_bg):
    config_dict = json.loads(requests.get(size_config_url).text)

    backbone_config = {}
    for stage_ix in range(1, 5):
        stage_config = config_dict[f"stage{stage_ix}"]

        merged_dict = {}

        # Iterate through the list of dictionaries
        for layer in stage_config:
            for key, value in layer.items():
                if key != "name":
                    # Check if the key is already in the merged_dict
                    if key in merged_dict:
                        merged_dict[key].append(value)
                    else:
                        # If the key is not in merged_dict, create a new list with the value
                        merged_dict[key] = [value]
        backbone_config[f"stage{stage_ix}"] = merged_dict

    neck_in_channels = []
    neck_out_channels = []
    neck_kernel_size = []
    neck_stride = []
    neck_dilation = []
    neck_groups = []

    for i in range(1, 5):
        layer_key = f"reduce_layer{i}"
        layer_dict = config_dict["neck"].get(layer_key)

        if layer_dict:
            # Append values to the corresponding lists
            neck_in_channels.append(layer_dict["in_channels"])
            neck_out_channels.append(layer_dict["out_channels"])
            neck_kernel_size.append(layer_dict["kernel_size"])
            neck_stride.append(layer_dict["stride"])
            neck_dilation.append(layer_dict["dilation"])
            neck_groups.append(layer_dict["groups"])

    return FastConfig(
        backbone_kernel_size=config_dict["first_conv"]["kernel_size"],
        backbone_stride=config_dict["first_conv"]["stride"],
        backbone_dilation=config_dict["first_conv"]["dilation"],
        backbone_groups=config_dict["first_conv"]["groups"],
        backbone_bias=config_dict["first_conv"]["bias"],
        backbone_has_shuffle=config_dict["first_conv"]["has_shuffle"],
        backbone_in_channels=config_dict["first_conv"]["in_channels"],
        backbone_out_channels=config_dict["first_conv"]["out_channels"],
        backbone_use_bn=config_dict["first_conv"]["use_bn"],
        backbone_act_func=config_dict["first_conv"]["act_func"],
        backbone_dropout_rate=config_dict["first_conv"]["dropout_rate"],
        backbone_ops_order=config_dict["first_conv"]["ops_order"],
        backbone_stage1_in_channels=backbone_config["stage1"]["in_channels"],
        backbone_stage1_out_channels=backbone_config["stage1"]["out_channels"],
        backbone_stage1_kernel_size=backbone_config["stage1"]["kernel_size"],
        backbone_stage1_stride=backbone_config["stage1"]["stride"],
        backbone_stage1_dilation=backbone_config["stage1"]["dilation"],
        backbone_stage1_groups=backbone_config["stage1"]["groups"],
        backbone_stage2_in_channels=backbone_config["stage2"]["in_channels"],
        backbone_stage2_out_channels=backbone_config["stage2"]["out_channels"],
        backbone_stage2_kernel_size=backbone_config["stage2"]["kernel_size"],
        backbone_stage2_stride=backbone_config["stage2"]["stride"],
        backbone_stage2_dilation=backbone_config["stage2"]["dilation"],
        backbone_stage2_groups=backbone_config["stage2"]["groups"],
        backbone_stage3_in_channels=backbone_config["stage3"]["in_channels"],
        backbone_stage3_out_channels=backbone_config["stage3"]["out_channels"],
        backbone_stage3_kernel_size=backbone_config["stage3"]["kernel_size"],
        backbone_stage3_stride=backbone_config["stage3"]["stride"],
        backbone_stage3_dilation=backbone_config["stage3"]["dilation"],
        backbone_stage3_groups=backbone_config["stage3"]["groups"],
        backbone_stage4_in_channels=backbone_config["stage4"]["in_channels"],
        backbone_stage4_out_channels=backbone_config["stage4"]["out_channels"],
        backbone_stage4_kernel_size=backbone_config["stage4"]["kernel_size"],
        backbone_stage4_stride=backbone_config["stage4"]["stride"],
        backbone_stage4_dilation=backbone_config["stage4"]["dilation"],
        backbone_stage4_groups=backbone_config["stage4"]["groups"],
        neck_in_channels=neck_in_channels,
        neck_out_channels=neck_out_channels,
        neck_kernel_size=neck_kernel_size,
        neck_stride=neck_stride,
        neck_dilation=neck_dilation,
        neck_groups=neck_groups,
        head_pooling_size=pooling_size,
        head_dropout_ratio=0.1,
        head_conv_in_channels=config_dict["head"]["conv"]["in_channels"],
        head_conv_out_channels=config_dict["head"]["conv"]["out_channels"],
        head_conv_kernel_size=config_dict["head"]["conv"]["kernel_size"],
        head_conv_stride=config_dict["head"]["conv"]["stride"],
        head_conv_dilation=config_dict["head"]["conv"]["dilation"],
        head_conv_groups=config_dict["head"]["conv"]["groups"],
        head_final_kernel_size=config_dict["head"]["final"]["kernel_size"],
        head_final_stride=config_dict["head"]["final"]["stride"],
        head_final_dilation=config_dict["head"]["final"]["dilation"],
        head_final_groups=config_dict["head"]["final"]["groups"],
        head_final_bias=config_dict["head"]["final"]["bias"],
        head_final_has_shuffle=config_dict["head"]["final"]["has_shuffle"],
        head_final_in_channels=config_dict["head"]["final"]["in_channels"],
        head_final_out_channels=config_dict["head"]["final"]["out_channels"],
        head_final_use_bn=config_dict["head"]["final"]["use_bn"],
        head_final_act_func=config_dict["head"]["final"]["act_func"],
        head_final_dropout_rate=config_dict["head"]["final"]["dropout_rate"],
        head_final_ops_order=config_dict["head"]["final"]["ops_order"],
        min_area=min_area,
        bbox_type=bbox_type,
        loss_bg=loss_bg,
    )


def get_small_model_config():
    pass


def get_base_model_config():
    pass


def convert_fast_checkpoint(checkpoint_url, checkpoint_config_url, pytorch_dump_folder_path, validate_logits):
    response = requests.get(checkpoint_config_url)
    content = response.text

    namespace = {}

    exec(content, namespace)

    model_config = namespace.get("model")
    test_config = namespace.get("test_cfg", None)
    data_config = namespace.get("data")

    min_area = 250
    bbox_type = "rect"
    loss_bg = False
    if test_config is not None:
        min_area = test_config.get("min_area", min_area)
        bbox_type = test_config.get("bbox_type", bbox_type)
        loss_bg = test_config.get("loss_emb", None) == "EmbLoss_v2"

    if "tiny" in model_config["backbone"]["config"]:
        config = prepare_config(
            tiny_config_url, model_config["detection_head"]["pooling_size"], min_area, bbox_type, loss_bg
        )
    elif "small" in model_config["backbone"]["config"]:
        config = prepare_config(
            small_config_url, model_config["detection_head"]["pooling_size"], min_area, bbox_type, loss_bg
        )
    else:
        config = prepare_config(
            base_config_url, model_config["detection_head"]["pooling_size"], min_area, bbox_type, loss_bg
        )
    size = 640
    if "train" in data_config:
        if "short_size" in data_config["train"]:
            size = data_config["train"]["short_size"]

    model = FastForSceneTextRecognition(config)
    fast_image_processor = FastImageProcessor(
        size={"height": size, "width": size},
        min_area=config.min_area,
        bbox_type=config.bbox_type,
        pooling_size=config.head_pooling_size,
    )
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", check_hash=True)["ema"]
    state_dict_changed = copy.deepcopy(state_dict)
    for key in state_dict:
        val = state_dict_changed.pop(key)
        new_key = key.replace("module.", "")
        for search, replacement in rename_key_mappings.items():
            if search in new_key:
                new_key = new_key.replace(search, replacement)
        state_dict_changed[new_key] = val
    model.load_state_dict(state_dict_changed)

    model.save_pretrained(pytorch_dump_folder_path)
    fast_image_processor.save_pretrained(pytorch_dump_folder_path)
    logging.info("The converted weights are save here : " + pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--checkpoint_config_url",
        default="https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    parser.add_argument(
        "--validate_logits",
        default=False,
        type=bool,
        help="whether to assert logits outputs",
    )
    args = parser.parse_args()

    convert_fast_checkpoint(
        args.checkpoint_url, args.checkpoint_config_url, args.pytorch_dump_folder_path, args.validate_logits
    )
