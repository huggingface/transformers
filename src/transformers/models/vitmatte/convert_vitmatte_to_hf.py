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
"""Convert VitMatte checkpoints from the original repository.

URL: https://github.com/hustvl/ViTMatte
"""

import argparse

import torch
from huggingface_hub import hf_hub_download

from transformers import VitDetConfig, VitMatteConfig, VitMatteForImageMatting


def get_config(model_name):
    backbone_config = VitDetConfig(out_features=["stage4"])

    return VitMatteConfig(backbone_config=backbone_config)


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

    # TODO decode head
    # fmt: on

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def convert_vitmatte_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    config = get_config(model_name)

    # load original state dict
    filepath = hf_hub_download(repo_id="nielsr/vitmatte-checkpoints", filename="ViTMatte_S_Com.pth", repo_type="model")
    state_dict = torch.load(filepath, map_location="cpu")

    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if "bn" in key:
            key = key.replace("bn", "batch_norm")
        state_dict[key] = val

    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # create model
    model = VitMatteForImageMatting(config)
    model.eval()

    # load state dict
    model.load_state_dict(state_dict)

    # TODO verify on image
    # url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
    # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # processor = SegformerImageProcessor()
    # pixel_values = processor(image, return_tensors="pt").pixel_values

    # with torch.no_grad():
    #     outputs = model(pixel_values)

    # print("Logits:", outputs.logits[0, 0, :3, :3])
    # assert torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4)
    # print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # print(f"Saving processor to {pytorch_dump_folder_path}")
        # processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and processor for {model_name} to hub")
        model.push_to_hub(f"openmmlab/{model_name}")
        # processor.push_to_hub(f"openmmlab/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="vitmatte-small-composition-1k",
        type=str,
        choices=["vitmatte-small-composition-1k"],
        help="Name of the VitMatte model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_vitmatte_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
