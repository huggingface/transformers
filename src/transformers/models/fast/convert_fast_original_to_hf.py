# coding=utf-8
# Copyright 2025 the Fast authors and The HuggingFace Inc. team. All rights reserved.
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
import json
import logging
import re
from collections import OrderedDict

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import FastConfig, FastForSceneTextRecognition, TextNetConfig
from transformers.models.fast.image_processing_fast import FastImageProcessor


tiny_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_tiny.config"
small_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_small.config"
base_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_base.config"

ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"module.backbone":                                         r"backbone.textnet",
    r"first_conv":                                              r"stem",
    r"bn":                                                      r"batch_norm",
    r"ver":                                                     r"vertical",
    r"hor":                                                     r"horizontal",
    r"module.neck":                                             r"neck",
    r"module.det_head":                                         r"text_detection_head",

    r"neck.reduce_layer1":                                      r"neck.reduce_layers.0",
    r"neck.reduce_layer2":                                      r"neck.reduce_layers.1",
    r"neck.reduce_layer3":                                      r"neck.reduce_layers.2",
    r"neck.reduce_layer4":                                      r"neck.reduce_layers.3",

    r"final.conv.weight":                                       r"final_conv.weight",
    r"neck.reduce_layers.1.rbr_identity.weight":                r"neck.reduce_layers.1.identity.weight",
    r"neck.reduce_layers.1.rbr_identity.bias":                  r"neck.reduce_layers.1.identity.bias",
    r"neck.reduce_layers.1.rbr_identity.running_mean":          r"neck.reduce_layers.1.identity.running_mean",
    r"neck.reduce_layers.1.rbr_identity.running_var":           r"neck.reduce_layers.1.identity.running_var",
    r"neck.reduce_layers.1.rbr_identity.num_batches_tracked":   r"neck.reduce_layers.1.identity.num_batches_tracked",

    r"textnet.stage1":                                          r"textnet.encoder.stages.0.stage",
    r"textnet.stage2":                                          r"textnet.encoder.stages.1.stage",
    r"textnet.stage3":                                          r"textnet.encoder.stages.2.stage",
    r"textnet.stage4":                                          r"textnet.encoder.stages.3.stage",
}

def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def get_model_config(model_config, model_type, size, min_area, bounding_box_type, loss_bg):
    model_config_map = {
        "tiny": {
            "config_url": tiny_config_url,
            "expected_logits": torch.tensor([-9.9181, -13.0701, -12.5045, -12.6523]),
            "expected_boxes": [(151, 151), (160, 56), (355, 74), (346, 169)],
        },
        "small": {
            "config_url": small_config_url,
            "expected_logits": torch.tensor([-13.1852, -17.2011, -16.9553, -16.8269]),
            "expected_boxes": [(154, 151), (155, 61), (351, 63), (350, 153)],
        },
        "base": {
            "config_url": base_config_url,
            "expected_logits": torch.tensor([-28.7481, -34.1635, -25.7430, -22.0260]),
            "expected_boxes": [(157, 149), (158, 66), (348, 68), (347, 151)],
        },
    }

    if model_type not in model_config_map:
        raise ValueError(f"Unknown model type: {model_type}")

    logits_config = model_config_map[model_type]
    config = prepare_config(
        logits_config["config_url"],
        size,
        model_config["detection_head"]["pooling_size"],
        min_area,
        bounding_box_type,
        loss_bg,
    )

    return config, logits_config["expected_logits"], logits_config["expected_boxes"]


def prepare_config(size_config_url, size, pooling_size, min_area, bounding_box_type, loss_bg):
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

    textnet_config = TextNetConfig(
        stem_kernel_size=config_dict["first_conv"]["kernel_size"],
        stem_stride=config_dict["first_conv"]["stride"],
        stem_num_channels=config_dict["first_conv"]["in_channels"],
        stem_out_channels=config_dict["first_conv"]["out_channels"],
        stem_act_func=config_dict["first_conv"]["act_func"],
        conv_layer_kernel_sizes=[
            backbone_config["stage1"]["kernel_size"],
            backbone_config["stage2"]["kernel_size"],
            backbone_config["stage3"]["kernel_size"],
            backbone_config["stage4"]["kernel_size"],
        ],
        conv_layer_strides=[
            backbone_config["stage1"]["stride"],
            backbone_config["stage2"]["stride"],
            backbone_config["stage3"]["stride"],
            backbone_config["stage4"]["stride"],
        ],
        hidden_sizes=[
            config_dict["first_conv"]["out_channels"],
            backbone_config["stage1"]["out_channels"][-1],
            backbone_config["stage2"]["out_channels"][-1],
            backbone_config["stage3"]["out_channels"][-1],
            backbone_config["stage4"]["out_channels"][-1],
        ],
        out_features=["stage1", "stage2", "stage3", "stage4"],
        out_indices=[1, 2, 3, 4],
    )

    fast_config = FastConfig(
        use_timm_backbone=False,
        backbone_config=textnet_config,
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
        bounding_box_type=bounding_box_type,
        loss_bg=loss_bg,
    )

    return fast_config


def convert_fast_checkpoint(
    checkpoint_url, checkpoint_config_filename, pytorch_dump_folder_path, save_backbone_separately, verify_logits
):
    config_filepath = hf_hub_download(repo_id="Raghavan/fast_model_config_files", filename="fast_model_configs.json")
    # we download the json file for safety reasons
    checkpoint_config_filename_to_json = checkpoint_config_filename.replace(".py", ".json")
    config_model_file_path = hf_hub_download(
        repo_id="jadechoghari/fast-configs", filename=checkpoint_config_filename_to_json
    )

    with open(config_filepath) as f:
        content = json.loads(f.read())

    with open(config_model_file_path) as f:
        content_model = json.loads(f.read())

    size = content[checkpoint_config_filename]["short_size"]
    model_config = content_model["model"]
    test_config = content_model.get("test_cfg", None)
    data_config = content_model["data"]
    min_area = 250
    bounding_box_type = "boxes"
    if test_config is not None:
        min_area = test_config.get("min_area", min_area)
        bbox_type = test_config.get("bounding_box_type", bounding_box_type)
        loss_bg = test_config.get("loss_emb", None) == "EmbLoss_v2"

    if bbox_type == "rect":
        bounding_box_type = "boxes"
    elif bbox_type == "poly":
        bounding_box_type = "polygons"
    else:
        bounding_box_type = bbox_type

    # determine model type from content
    model_type = None
    for key in ["tiny", "small", "base"]:
        if key in content[checkpoint_config_filename]["config"]:
            model_type = key
            break

    if model_type is None:
        raise ValueError("Model type not found in checkpoint config.")

    # get model config
    config, expected_slice_logits, expected_slice_boxes = get_model_config(
        model_config, model_type, size, min_area, bounding_box_type, loss_bg
    )
    size = data_config.get("train", {}).get("short_size", size)
    model = FastForSceneTextRecognition(config)
    fast_image_processor = FastImageProcessor(
        size={"shortest_edge": size},
        min_area=config.min_area,
        bounding_box_type=config.bounding_box_type,
        pooling_size=config.head_pooling_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location=device, check_hash=True)["ema"]
    state_dict_changed = OrderedDict()
    
    old_keys = list(state_dict.keys())
    new_key_mapping = convert_old_keys_to_new_keys(old_keys)

    for key, value in state_dict.items():
        new_key = new_key_mapping.get(key, key)          
        if new_key == "":                  
            continue

        state_dict_changed[new_key] = value

    model.load_state_dict(state_dict_changed)
    model.eval()

    if verify_logits:
        url = "https://huggingface.co/datasets/Raghavan/fast_model_samples/resolve/main/img657.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        pixel_values = fast_image_processor(image, return_tensors="pt").pixel_values

        with torch.no_grad():
            output = model(pixel_values)

        # test the logits
        torch.testing.assert_close(output.logits[0][0][0][:4], expected_slice_logits, rtol=1e-4, atol=1e-4)
        target_sizes = [(image.height, image.width)]
        threshold = 0.88
        text_locations = fast_image_processor.post_process_text_detection(
            output, target_sizes, threshold
        )
        if text_locations[0]["boxes"][0] != expected_slice_boxes:
            raise ValueError(f"Expected {expected_slice_boxes}, but got {text_locations[0]['boxes'][0]}")

    model.save_pretrained(pytorch_dump_folder_path)
    if save_backbone_separately:
        model.backbone.save_pretrained(pytorch_dump_folder_path + "/textnet/")
    fast_image_processor.save_pretrained(pytorch_dump_folder_path)
    logging.info("The converted weights are saved here : " + pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/czczup/FAST/releases/download/release/fast_tiny_ic17mlt_640.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--checkpoint_config_filename",
        default="fast_tiny_ic17mlt_640.py",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--save_backbone_separately",
        default=True,
        type=bool,
        help="whether to assert logits outputs",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/home/user/app/transformers/src/transformers/models/fast/output",
        type=str,
        help="Path to the folder to output PyTorch model.",
    )
    parser.add_argument(
        "--verify_logits",
        action="store_false",
        required=False,
        help="Whether to verify the logits after conversion.",
    )
    args = parser.parse_args()

    convert_fast_checkpoint(
        args.checkpoint_url,
        args.checkpoint_config_filename,
        args.pytorch_dump_folder_path,
        args.save_backbone_separately,
        args.verify_logits,
    )
