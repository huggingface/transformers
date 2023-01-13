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
"""Convert ConvNext + UperNet checkpoints from mmsegmentation."""

import argparse
import json

import torch
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import ConvNextConfig, SegformerImageProcessor, UperNetConfig, UperNetForSemanticSegmentation


def get_upernet_config(model_name):
    auxiliary_in_channels = 384
    if "tiny" in model_name:
        depths = [3, 3, 9, 3]
        hidden_sizes = [96, 192, 384, 768]
    if "small" in model_name:
        depths = [3, 3, 27, 3]
        hidden_sizes = [96, 192, 384, 768]
    if "base" in model_name:
        depths = [3, 3, 27, 3]
        hidden_sizes = [128, 256, 512, 1024]
        auxiliary_in_channels = 512
    if "large" in model_name:
        depths = [3, 3, 27, 3]
        hidden_sizes = [192, 384, 768, 1536]
        auxiliary_in_channels = 768
    if "xlarge" in model_name:
        depths = [3, 3, 27, 3]
        hidden_sizes = [256, 512, 1024, 2048]
        auxiliary_in_channels = 1024

    # set label information
    num_labels = 150
    repo_id = "huggingface/label-files"
    filename = "ade20k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    backbone_config = ConvNextConfig(
        depths=depths, hidden_sizes=hidden_sizes, out_features=["stage1", "stage2", "stage3", "stage4"]
    )
    config = UperNetConfig(
        backbone_config=backbone_config,
        auxiliary_in_channels=auxiliary_in_channels,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    return config


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []

    # fmt: off
    # stem
    rename_keys.append(("backbone.downsample_layers.0.0.weight", "backbone.embeddings.patch_embeddings.weight"))
    rename_keys.append(("backbone.downsample_layers.0.0.bias", "backbone.embeddings.patch_embeddings.bias"))
    rename_keys.append(("backbone.downsample_layers.0.1.weight", "backbone.embeddings.layernorm.weight"))
    rename_keys.append(("backbone.downsample_layers.0.1.bias", "backbone.embeddings.layernorm.bias"))
    # stages
    for i in range(len(config.backbone_config.depths)):
        for j in range(config.backbone_config.depths[i]):
            rename_keys.append((f"backbone.stages.{i}.{j}.gamma", f"backbone.encoder.stages.{i}.layers.{j}.layer_scale_parameter"))
            rename_keys.append((f"backbone.stages.{i}.{j}.depthwise_conv.weight", f"backbone.encoder.stages.{i}.layers.{j}.dwconv.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.depthwise_conv.bias", f"backbone.encoder.stages.{i}.layers.{j}.dwconv.bias"))
            rename_keys.append((f"backbone.stages.{i}.{j}.norm.weight", f"backbone.encoder.stages.{i}.layers.{j}.layernorm.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.norm.bias", f"backbone.encoder.stages.{i}.layers.{j}.layernorm.bias"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv1.weight", f"backbone.encoder.stages.{i}.layers.{j}.pwconv1.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv1.bias", f"backbone.encoder.stages.{i}.layers.{j}.pwconv1.bias"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv2.weight", f"backbone.encoder.stages.{i}.layers.{j}.pwconv2.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv2.bias", f"backbone.encoder.stages.{i}.layers.{j}.pwconv2.bias"))
        if i > 0:
            rename_keys.append((f"backbone.downsample_layers.{i}.0.weight", f"backbone.encoder.stages.{i}.downsampling_layer.0.weight"))
            rename_keys.append((f"backbone.downsample_layers.{i}.0.bias", f"backbone.encoder.stages.{i}.downsampling_layer.0.bias"))
            rename_keys.append((f"backbone.downsample_layers.{i}.1.weight", f"backbone.encoder.stages.{i}.downsampling_layer.1.weight"))
            rename_keys.append((f"backbone.downsample_layers.{i}.1.bias", f"backbone.encoder.stages.{i}.downsampling_layer.1.bias"))

        rename_keys.append((f"backbone.norm{i}.weight", f"backbone.hidden_states_norms.stage{i+1}.weight"))
        rename_keys.append((f"backbone.norm{i}.bias", f"backbone.hidden_states_norms.stage{i+1}.bias"))

    # decode head
    rename_keys.extend(
        [
            ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
            ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
            ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
            ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
        ]
    )
    # fmt: on

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def convert_upernet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    model_name_to_url = {
        "upernet-convnext-tiny": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k/upernet_convnext_tiny_fp16_512x512_160k_ade20k_20220227_124553-cad485de.pth",
        "upernet-convnext-small": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_small_fp16_512x512_160k_ade20k/upernet_convnext_small_fp16_512x512_160k_ade20k_20220227_131208-1b1e394f.pth",
        "upernet-convnext-base": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_base_fp16_512x512_160k_ade20k/upernet_convnext_base_fp16_512x512_160k_ade20k_20220227_181227-02a24fc6.pth",
        "upernet-convnext-large": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_large_fp16_640x640_160k_ade20k/upernet_convnext_large_fp16_640x640_160k_ade20k_20220226_040532-e57aa54d.pth",
        "upernet-convnext-xlarge": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_xlarge_fp16_640x640_160k_ade20k/upernet_convnext_xlarge_fp16_640x640_160k_ade20k_20220226_080344-95fc38c2.pth",
    }
    checkpoint_url = model_name_to_url[model_name]
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["state_dict"]

    config = get_upernet_config(model_name)
    model = UperNetForSemanticSegmentation(config)
    model.eval()

    # replace "bn" => "batch_norm"
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if "bn" in key:
            key = key.replace("bn", "batch_norm")
        state_dict[key] = val

    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    model.load_state_dict(state_dict)

    # verify on image
    url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    processor = SegformerImageProcessor()
    pixel_values = processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model(pixel_values)

    if model_name == "upernet-convnext-tiny":
        expected_slice = torch.tensor(
            [[-8.8110, -8.8110, -8.6521], [-8.8110, -8.8110, -8.6521], [-8.7746, -8.7746, -8.6130]]
        )
    elif model_name == "upernet-convnext-small":
        expected_slice = torch.tensor(
            [[-8.8236, -8.8236, -8.6771], [-8.8236, -8.8236, -8.6771], [-8.7638, -8.7638, -8.6240]]
        )
    elif model_name == "upernet-convnext-base":
        expected_slice = torch.tensor(
            [[-8.8558, -8.8558, -8.6905], [-8.8558, -8.8558, -8.6905], [-8.7669, -8.7669, -8.6021]]
        )
    elif model_name == "upernet-convnext-large":
        expected_slice = torch.tensor(
            [[-8.6660, -8.6660, -8.6210], [-8.6660, -8.6660, -8.6210], [-8.6310, -8.6310, -8.5964]]
        )
    elif model_name == "upernet-convnext-xlarge":
        expected_slice = torch.tensor(
            [[-8.4980, -8.4980, -8.3977], [-8.4980, -8.4980, -8.3977], [-8.4379, -8.4379, -8.3412]]
        )
    print("Logits:", outputs.logits[0, 0, :3, :3])
    assert torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving processor to {pytorch_dump_folder_path}")
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and processor for {model_name} to hub")
        model.push_to_hub(f"openmmlab/{model_name}")
        processor.push_to_hub(f"openmmlab/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="upernet-convnext-tiny",
        type=str,
        choices=[f"upernet-convnext-{size}" for size in ["tiny", "small", "base", "large", "xlarge"]],
        help="Name of the ConvNext UperNet model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_upernet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
