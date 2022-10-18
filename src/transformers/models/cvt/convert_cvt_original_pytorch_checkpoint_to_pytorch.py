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
"""Convert CvT checkpoints from the original repository.

URL: https://github.com/microsoft/CvT"""


import argparse
import json
from collections import OrderedDict

import torch

from huggingface_hub import cached_download, hf_hub_url
from transformers import AutoFeatureExtractor, CvtConfig, CvtForImageClassification


def embeddings(idx):
    """
    The function helps in renaming embedding layer weights.

    Args:
        idx: stage number in original model
    """
    embed = []
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.projection.weight",
            f"stage{idx}.patch_embed.proj.weight",
        )
    )
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.projection.bias",
            f"stage{idx}.patch_embed.proj.bias",
        )
    )
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.normalization.weight",
            f"stage{idx}.patch_embed.norm.weight",
        )
    )
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.normalization.bias",
            f"stage{idx}.patch_embed.norm.bias",
        )
    )
    return embed


def attention(idx, cnt):
    """
    The function helps in renaming attention block layers weights.

    Args:
        idx: stage number in original model
        cnt: count of blocks in each stage
    """
    attention_weights = []
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.convolution.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.conv.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.bias",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.bias",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.running_mean",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.running_mean",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.running_var",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.running_var",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.num_batches_tracked",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.num_batches_tracked",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.convolution.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.conv.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.bias",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.bias",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.running_mean",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.running_mean",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.running_var",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.running_var",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.num_batches_tracked",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.num_batches_tracked",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.convolution.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.conv.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.bias",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.bias",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.running_mean",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.running_mean",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.running_var",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.running_var",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.num_batches_tracked",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.num_batches_tracked",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_query.weight",
            f"stage{idx}.blocks.{cnt}.attn.proj_q.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_query.bias",
            f"stage{idx}.blocks.{cnt}.attn.proj_q.bias",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_key.weight",
            f"stage{idx}.blocks.{cnt}.attn.proj_k.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_key.bias",
            f"stage{idx}.blocks.{cnt}.attn.proj_k.bias",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_value.weight",
            f"stage{idx}.blocks.{cnt}.attn.proj_v.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_value.bias",
            f"stage{idx}.blocks.{cnt}.attn.proj_v.bias",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.output.dense.weight",
            f"stage{idx}.blocks.{cnt}.attn.proj.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.output.dense.bias",
            f"stage{idx}.blocks.{cnt}.attn.proj.bias",
        )
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.intermediate.dense.weight", f"stage{idx}.blocks.{cnt}.mlp.fc1.weight")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.intermediate.dense.bias", f"stage{idx}.blocks.{cnt}.mlp.fc1.bias")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.output.dense.weight", f"stage{idx}.blocks.{cnt}.mlp.fc2.weight")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.output.dense.bias", f"stage{idx}.blocks.{cnt}.mlp.fc2.bias")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_before.weight", f"stage{idx}.blocks.{cnt}.norm1.weight")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_before.bias", f"stage{idx}.blocks.{cnt}.norm1.bias")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_after.weight", f"stage{idx}.blocks.{cnt}.norm2.weight")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_after.bias", f"stage{idx}.blocks.{cnt}.norm2.bias")
    )
    return attention_weights


def cls_token(idx):
    """
    Function helps in renaming cls_token weights
    """
    token = []
    token.append((f"cvt.encoder.stages.{idx}.cls_token", "stage2.cls_token"))
    return token


def final():
    """
    Function helps in renaming final classification layer
    """
    head = []
    head.append(("layernorm.weight", "norm.weight"))
    head.append(("layernorm.bias", "norm.bias"))
    head.append(("classifier.weight", "head.weight"))
    head.append(("classifier.bias", "head.bias"))
    return head


def convert_cvt_checkpoint(cvt_model, image_size, cvt_file_name, pytorch_dump_folder):
    """
    Fucntion to convert the microsoft cvt checkpoint to huggingface checkpoint
    """
    img_labels_file = "imagenet-1k-id2label.json"
    num_labels = 1000

    repo_id = "huggingface/label-files"
    num_labels = num_labels
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, img_labels_file, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}

    config = config = CvtConfig(num_labels=num_labels, id2label=id2label, label2id=label2id)

    # For depth size 13 (13 = 1+2+10)
    if cvt_model.rsplit("/", 1)[-1][4:6] == "13":
        config.depth = [1, 2, 10]

    # For depth size 21 (21 = 1+4+16)
    elif cvt_model.rsplit("/", 1)[-1][4:6] == "21":
        config.depth = [1, 4, 16]

    # For wide cvt (similar to wide-resnet) depth size 24 (w24 = 2 + 2 20)
    else:
        config.depth = [2, 2, 20]
        config.num_heads = [3, 12, 16]
        config.embed_dim = [192, 768, 1024]

    model = CvtForImageClassification(config)
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/convnext-base-224-22k-1k")
    feature_extractor.size = image_size
    original_weights = torch.load(cvt_file_name, map_location=torch.device("cpu"))

    huggingface_weights = OrderedDict()
    list_of_state_dict = []

    for idx in range(len(config.depth)):
        if config.cls_token[idx]:
            list_of_state_dict = list_of_state_dict + cls_token(idx)
        list_of_state_dict = list_of_state_dict + embeddings(idx)
        for cnt in range(config.depth[idx]):
            list_of_state_dict = list_of_state_dict + attention(idx, cnt)

    list_of_state_dict = list_of_state_dict + final()
    for gg in list_of_state_dict:
        print(gg)
    for i in range(len(list_of_state_dict)):
        huggingface_weights[list_of_state_dict[i][0]] = original_weights[list_of_state_dict[i][1]]

    model.load_state_dict(huggingface_weights)
    model.save_pretrained(pytorch_dump_folder)
    feature_extractor.save_pretrained(pytorch_dump_folder)


# Download the weights from zoo: https://1drv.ms/u/s!AhIXJn_J-blW9RzF3rMW7SsLHa8h?e=blQ0Al

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cvt_model",
        default="cvt-w24",
        type=str,
        help="Name of the cvt model you'd like to convert.",
    )
    parser.add_argument(
        "--image_size",
        default=384,
        type=int,
        help="Input Image Size",
    )
    parser.add_argument(
        "--cvt_file_name",
        default="cvtmodels\CvT-w24-384x384-IN-22k.pth",
        type=str,
        help="Input Image Size",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_cvt_checkpoint(args.cvt_model, args.image_size, args.cvt_file_name, args.pytorch_dump_folder_path)
