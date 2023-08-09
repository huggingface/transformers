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
"""Convert MobileViTV2 checkpoints from the ml-cvnets library."""


import argparse
import collections
import json
from pathlib import Path

import requests
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    MobileViTImageProcessor,
    MobileViTV2Config,
    MobileViTV2ForImageClassification,
    MobileViTV2ForSemanticSegmentation,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_orig_config_file(orig_cfg_file):
    print("Loading config file...")

    def flatten_yaml_as_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.abc.MutableMapping):
                items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    config = argparse.Namespace()
    with open(orig_cfg_file, "r") as yaml_file:
        try:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

            flat_cfg = flatten_yaml_as_dict(cfg)
            for k, v in flat_cfg.items():
                setattr(config, k, v)
        except yaml.YAMLError as exc:
            logger.error("Error while loading config file: {}. Error message: {}".format(orig_cfg_file, str(exc)))
    return config


def get_mobilevitv2_config(task_name, orig_cfg_file):
    config = MobileViTV2Config()

    is_segmentation_model = False

    # dataset
    if task_name.startswith("imagenet1k_"):
        config.num_labels = 1000
        if int(task_name.strip().split("_")[-1]) == 384:
            config.image_size = 384
        else:
            config.image_size = 256
        filename = "imagenet-1k-id2label.json"
    elif task_name.startswith("imagenet21k_to_1k_"):
        config.num_labels = 21000
        if int(task_name.strip().split("_")[-1]) == 384:
            config.image_size = 384
        else:
            config.image_size = 256
        filename = "imagenet-22k-id2label.json"
    elif task_name.startswith("ade20k_"):
        config.num_labels = 151
        config.image_size = 512
        filename = "ade20k-id2label.json"
        is_segmentation_model = True
    elif task_name.startswith("voc_"):
        config.num_labels = 21
        config.image_size = 512
        filename = "pascal-voc-id2label.json"
        is_segmentation_model = True

    # orig_config
    orig_config = load_orig_config_file(orig_cfg_file)
    assert getattr(orig_config, "model.classification.name", -1) == "mobilevit_v2", "Invalid model"
    config.width_multiplier = getattr(orig_config, "model.classification.mitv2.width_multiplier", 1.0)
    assert (
        getattr(orig_config, "model.classification.mitv2.attn_norm_layer", -1) == "layer_norm_2d"
    ), "Norm layers other than layer_norm_2d is not supported"
    config.hidden_act = getattr(orig_config, "model.classification.activation.name", "swish")
    # config.image_size == getattr(orig_config,  'sampler.bs.crop_size_width', 256)

    if is_segmentation_model:
        config.output_stride = getattr(orig_config, "model.segmentation.output_stride", 16)
        if "_deeplabv3" in task_name:
            config.atrous_rates = getattr(orig_config, "model.segmentation.deeplabv3.aspp_rates", [12, 24, 36])
            config.aspp_out_channels = getattr(orig_config, "model.segmentation.deeplabv3.aspp_out_channels", 512)
            config.aspp_dropout_prob = getattr(orig_config, "model.segmentation.deeplabv3.aspp_dropout", 0.1)

    # id2label
    repo_id = "huggingface/label-files"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def create_rename_keys(state_dict, base_model=False):
    if base_model:
        model_prefix = ""
    else:
        model_prefix = "mobilevitv2."

    rename_keys = []
    for k in state_dict.keys():
        if k[:8] == "encoder.":
            k_new = k[8:]
        else:
            k_new = k

        if ".block." in k:
            k_new = k_new.replace(".block.", ".")
        if ".conv." in k:
            k_new = k_new.replace(".conv.", ".convolution.")
        if ".norm." in k:
            k_new = k_new.replace(".norm.", ".normalization.")

        if "conv_1." in k:
            k_new = k_new.replace("conv_1.", f"{model_prefix}conv_stem.")
        for i in [1, 2]:
            if f"layer_{i}." in k:
                k_new = k_new.replace(f"layer_{i}.", f"{model_prefix}encoder.layer.{i-1}.layer.")
        if ".exp_1x1." in k:
            k_new = k_new.replace(".exp_1x1.", ".expand_1x1.")
        if ".red_1x1." in k:
            k_new = k_new.replace(".red_1x1.", ".reduce_1x1.")

        for i in [3, 4, 5]:
            if f"layer_{i}.0." in k:
                k_new = k_new.replace(f"layer_{i}.0.", f"{model_prefix}encoder.layer.{i-1}.downsampling_layer.")
            if f"layer_{i}.1.local_rep.0." in k:
                k_new = k_new.replace(f"layer_{i}.1.local_rep.0.", f"{model_prefix}encoder.layer.{i-1}.conv_kxk.")
            if f"layer_{i}.1.local_rep.1." in k:
                k_new = k_new.replace(f"layer_{i}.1.local_rep.1.", f"{model_prefix}encoder.layer.{i-1}.conv_1x1.")

        for i in [3, 4, 5]:
            if i == 3:
                j_in = [0, 1]
            elif i == 4:
                j_in = [0, 1, 2, 3]
            elif i == 5:
                j_in = [0, 1, 2]

            for j in j_in:
                if f"layer_{i}.1.global_rep.{j}." in k:
                    k_new = k_new.replace(
                        f"layer_{i}.1.global_rep.{j}.", f"{model_prefix}encoder.layer.{i-1}.transformer.layer.{j}."
                    )
            if f"layer_{i}.1.global_rep.{j+1}." in k:
                k_new = k_new.replace(
                    f"layer_{i}.1.global_rep.{j+1}.", f"{model_prefix}encoder.layer.{i-1}.layernorm."
                )

            if f"layer_{i}.1.conv_proj." in k:
                k_new = k_new.replace(f"layer_{i}.1.conv_proj.", f"{model_prefix}encoder.layer.{i-1}.conv_projection.")

        if "pre_norm_attn.0." in k:
            k_new = k_new.replace("pre_norm_attn.0.", "layernorm_before.")
        if "pre_norm_attn.1." in k:
            k_new = k_new.replace("pre_norm_attn.1.", "attention.")
        if "pre_norm_ffn.0." in k:
            k_new = k_new.replace("pre_norm_ffn.0.", "layernorm_after.")
        if "pre_norm_ffn.1." in k:
            k_new = k_new.replace("pre_norm_ffn.1.", "ffn.conv1.")
        if "pre_norm_ffn.3." in k:
            k_new = k_new.replace("pre_norm_ffn.3.", "ffn.conv2.")

        if "classifier.1." in k:
            k_new = k_new.replace("classifier.1.", "classifier.")

        if "seg_head." in k:
            k_new = k_new.replace("seg_head.", "segmentation_head.")
        if ".aspp_layer." in k:
            k_new = k_new.replace(".aspp_layer.", ".")
        if ".aspp_pool." in k:
            k_new = k_new.replace(".aspp_pool.", ".")

        rename_keys.append((k, k_new))
    return rename_keys


def remove_unused_keys(state_dict):
    """remove unused keys (e.g.: seg_head.aux_head)"""
    keys_to_ignore = []
    for k in state_dict.keys():
        if k.startswith("seg_head.aux_head."):
            keys_to_ignore.append(k)
    for k in keys_to_ignore:
        state_dict.pop(k, None)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # url = "https://cdn.britannica.com/86/141086-050-9D7C75EE/Gulfstream-G450-business-jet-passengers.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_mobilevitv2_checkpoint(task_name, checkpoint_path, orig_config_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our MobileViTV2 structure.
    """
    config = get_mobilevitv2_config(task_name, orig_config_path)

    # load original state_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # load huggingface model
    if task_name.startswith("ade20k_") or task_name.startswith("voc_"):
        model = MobileViTV2ForSemanticSegmentation(config).eval()
        base_model = False
    else:
        model = MobileViTV2ForImageClassification(config).eval()
        base_model = False

    # remove and rename some keys of load the original model
    state_dict = checkpoint
    remove_unused_keys(state_dict)
    rename_keys = create_rename_keys(state_dict, base_model=base_model)
    for rename_key_src, rename_key_dest in rename_keys:
        rename_key(state_dict, rename_key_src, rename_key_dest)

    # load modified state_dict
    model.load_state_dict(state_dict)

    # Check outputs on an image, prepared by MobileViTImageProcessor
    image_processor = MobileViTImageProcessor(crop_size=config.image_size, size=config.image_size + 32)
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    outputs = model(**encoding)

    # verify classification model
    if task_name.startswith("imagenet"):
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", model.config.id2label[predicted_class_idx])
        if task_name.startswith("imagenet1k_256") and config.width_multiplier == 1.0:
            # expected_logits for base variant
            expected_logits = torch.tensor([-1.6336e00, -7.3204e-02, -5.1883e-01])
            assert torch.allclose(logits[0, :3], expected_logits, atol=1e-4)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {task_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--task",
        default="imagenet1k_256",
        type=str,
        help=(
            "Name of the task for which the MobileViTV2 model you'd like to convert is trained on . "
            """
                Classification (ImageNet-1k)
                    - MobileViTV2 (256x256) : imagenet1k_256
                    - MobileViTV2 (Trained on 256x256 and Finetuned on 384x384) : imagenet1k_384
                    - MobileViTV2 (Trained on ImageNet-21k and Finetuned on ImageNet-1k 256x256) :
                      imagenet21k_to_1k_256
                    - MobileViTV2 (Trained on ImageNet-21k, Finetuned on ImageNet-1k 256x256, and Finetuned on
                      ImageNet-1k 384x384) : imagenet21k_to_1k_384
                Segmentation
                    - ADE20K Dataset : ade20k_deeplabv3
                    - Pascal VOC 2012 Dataset: voc_deeplabv3
            """
        ),
        choices=[
            "imagenet1k_256",
            "imagenet1k_384",
            "imagenet21k_to_1k_256",
            "imagenet21k_to_1k_384",
            "ade20k_deeplabv3",
            "voc_deeplabv3",
        ],
    )

    parser.add_argument(
        "--orig_checkpoint_path", required=True, type=str, help="Path to the original state dict (.pt file)."
    )
    parser.add_argument("--orig_config_path", required=True, type=str, help="Path to the original config file.")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_mobilevitv2_checkpoint(
        args.task, args.orig_checkpoint_path, args.orig_config_path, args.pytorch_dump_folder_path
    )
