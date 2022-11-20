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

import torch

from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation


def get_upernet_config(model_name):
    backbone_config = ConvNextConfig()
    config = UperNetConfig(backbone_config=backbone_config)

    return config


def rename_key(name):
    if "encoder.mask_token" in name:
        name = name.replace("encoder.mask_token", "embeddings.mask_token")

    return name


def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "attn_mask" in key:
            pass
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_upernet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    model_name_to_url = {
        "convnext-tiny-upernet": "https://download.openmmlab.com/mmsegmentation/v0.5/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k/upernet_convnext_tiny_fp16_512x512_160k_ade20k_20220227_124553-cad485de.pth",
    }
    checkpoint_url = model_name_to_url[model_name]
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["state_dict"]

    config = get_upernet_config(model_name)
    model = UperNetForSemanticSegmentation(config)
    model.eval()

    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # TODO verify on image
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # eature_extractor = ViTFeatureExtractor(size={"height": 192, "width": 192})
    # image = Image.open(requests.get(url, stream=True).raw)
    # inputs = feature_extractor(images=image, return_tensors="pt")

    # with torch.no_grad():
    #     outputs = model(**inputs).logits

    # print(outputs.keys())
    # print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

        # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and feature extractor for {model_name} to hub")
        model.push_to_hub(f"microsoft/{model_name}")
        # feature_extractor.push_to_hub(f"microsoft/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="convnext-tiny-upernet",
        type=str,
        choices=["convnext-tiny-upernet", "convnext-base-upernet"],
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
