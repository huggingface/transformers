# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert SegFormer checkpoints."""


import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import cached_download, hf_hub_url
from transformers import (
    SegformerConfig,
    SegformerFeatureExtractor,
    SegformerForImageClassification,
    SegformerForImageSegmentation,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_keys(state_dict, encoder_only=False):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if encoder_only and not key.startswith("head"):
            key = "segformer.encoder." + key
        if key.startswith("backbone"):
            key = key.replace("backbone", "segformer.encoder")
        if "patch_embed" in key:
            # replace for example patch_embed1 by patch_embeddings.0
            idx = key[key.find("patch_embed") + len("patch_embed")]
            key = key.replace(f"patch_embed{idx}", f"patch_embeddings.{int(idx)-1}")
        if "norm" in key:
            key = key.replace("norm", "layer_norm")
        if "segformer.encoder.layer_norm" in key:
            # replace for example layer_norm1 by layer_norm.0
            idx = key[key.find("segformer.encoder.layer_norm") + len("segformer.encoder.layer_norm")]
            key = key.replace(f"layer_norm{idx}", f"layer_norm.{int(idx)-1}")
        if "layer_norm1" in key:
            key = key.replace("layer_norm1", "layer_norm_1")
        if "layer_norm2" in key:
            key = key.replace("layer_norm2", "layer_norm_2")
        if "block" in key:
            # replace for example block1 by block.0
            idx = key[key.find("block") + len("block")]
            key = key.replace(f"block{idx}", f"block.{int(idx)-1}")
        if "attn.q" in key:
            key = key.replace("attn.q", "attention.self.query")
        if "attn.proj" in key:
            key = key.replace("attn.proj", "attention.output.dense")
        if "attn" in key:
            key = key.replace("attn", "attention.self")
        if "fc1" in key:
            key = key.replace("fc1", "dense1")
        if "fc2" in key:
            key = key.replace("fc2", "dense2")
        if "linear_pred" in key:
            key = key.replace("linear_pred", "classifier")
        if "linear_fuse" in key:
            key = key.replace("linear_fuse.conv", "linear_fuse")
            key = key.replace("linear_fuse.bn", "batch_norm")
        if "linear_c" in key:
            # replace for example linear_c4 by linear_c.3
            idx = key[key.find("linear_c") + len("linear_c")]
            key = key.replace(f"linear_c{idx}", f"linear_c.{int(idx)-1}")
        if key.startswith("head"):
            key = key.replace("head", "classifier")
        new_state_dict[key] = value

    return new_state_dict


def read_in_k_v(state_dict, config):
    # for each of the encoder blocks:
    for i in range(config.num_encoder_blocks):
        for j in range(config.depths[i]):
            # read in weights + bias of keys and values (which is a single matrix in the original implementation)
            kv_weight = state_dict.pop(f"segformer.encoder.block.{i}.{j}.attention.self.kv.weight")
            kv_bias = state_dict.pop(f"segformer.encoder.block.{i}.{j}.attention.self.kv.bias")
            # next, add keys and values (in that order) to the state dict
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.key.weight"] = kv_weight[
                : config.hidden_sizes[i], :
            ]
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.key.bias"] = kv_bias[: config.hidden_sizes[i]]
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.value.weight"] = kv_weight[
                config.hidden_sizes[i] :, :
            ]
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.value.bias"] = kv_bias[
                config.hidden_sizes[i] :
            ]


# We will verify our results on a COCO image
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    return image


@torch.no_grad()
def convert_segformer_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our SegFormer structure.
    """

    # load default SegFormer configuration
    config = SegformerConfig()
    encoder_only = False

    # set attributes based on model_name
    repo_id = "datasets/huggingface/label-files"
    if "segformer" in model_name:
        size = model_name[len("segformer.") : len("segformer.") + 2]
        if "ade" in model_name:
            config.num_labels = 150
            filename = "ade20k-id2label.json"
        elif "city" in model_name:
            config.num_labels = 19
            filename = "cityscapes-id2label.json"
        else:
            raise ValueError(f"Model {model_name} not supported")
    elif "mit" in model_name:
        encoder_only = True
        size = model_name[4:6]
        config.num_labels = 1000
        filename = "imagenet-1k-id2label.json"
    else:
        raise ValueError(f"Model {model_name} not supported")

    # set config attributes
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename)), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    if size == "b0":
        pass
    elif size == "b1":
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 256
    elif size == "b2":
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 768
        config.depths = [3, 4, 6, 3]
    elif size == "b3":
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 768
        config.depths = [3, 4, 18, 3]
    elif size == "b4":
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 768
        config.depths = [3, 8, 27, 3]
    elif size == "b5":
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 768
        config.depths = [3, 6, 40, 3]
    else:
        raise ValueError(f"Size {size} not supported")

    # load feature extractor (only resize + normalize)
    feature_extractor = SegformerFeatureExtractor(
        image_scale=(512, 512), keep_ratio=False, align=False, do_random_crop=False
    )

    # prepare image
    image = prepare_img()
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    logger.info(f"Converting model {model_name}...")

    # load original state dict
    if encoder_only:
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    else:
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))["state_dict"]

    # rename keys
    state_dict = rename_keys(state_dict, encoder_only=encoder_only)
    if not encoder_only:
        del state_dict["decode_head.conv_seg.weight"]
        del state_dict["decode_head.conv_seg.bias"]

    # key and value matrices need special treatment
    read_in_k_v(state_dict, config)

    # create HuggingFace model and load state dict
    if encoder_only:
        model = SegformerForImageClassification(config)
    else:
        model = SegformerForImageSegmentation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # forward pass
    outputs = model(pixel_values)

    # verify logits
    if not encoder_only:
        if size == "b0":
            logits = outputs.logits
            assert logits.shape == (1, 150, 128, 128)
            expected_slice = torch.tensor(
                [
                    [[-4.6310, -5.5232, -6.2356], [-5.1921, -6.1444, -6.5996], [-5.4424, -6.2790, -6.7574]],
                    [[-12.1391, -13.3122, -13.9554], [-12.8732, -13.9352, -14.3563], [-12.9438, -13.8226, -14.2513]],
                    [[-12.5134, -13.4686, -14.4915], [-12.8669, -14.4343, -14.7758], [-13.2523, -14.5819, -15.0694]],
                ]
            )
            assert torch.allclose(outputs.logits[0, :3, :3, :3], expected_slice, atol=1e-4)

    else:
        assert outputs.logits.shape == (1, 1000)

    # finally, save model and feature extractor
    logger.info(f"Saving PyTorch model and feature extractor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="segformer.b0.512x512.ade.160k",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the original PyTorch checkpoint (.pth file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_segformer_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path)
