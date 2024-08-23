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
"""Convert DETA checkpoints from the original repository.

URL: https://github.com/jozhang97/DETA/tree/master"""


import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import cached_download, hf_hub_download, hf_hub_url
from PIL import Image

from transformers import DetaConfig, DetaForObjectDetection, DetaImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_deta_config():
    config = DetaConfig(
        num_queries=900,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        num_feature_levels=5,
        assign_first_stage=True,
        with_box_refine=True,
        two_stage=True,
    )

    # set labels
    config.num_labels = 91
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []

    # stem
    # fmt: off
    rename_keys.append(("backbone.0.body.conv1.weight", "model.backbone.model.embedder.embedder.convolution.weight"))
    rename_keys.append(("backbone.0.body.bn1.weight", "model.backbone.model.embedder.embedder.normalization.weight"))
    rename_keys.append(("backbone.0.body.bn1.bias", "model.backbone.model.embedder.embedder.normalization.bias"))
    rename_keys.append(("backbone.0.body.bn1.running_mean", "model.backbone.model.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("backbone.0.body.bn1.running_var", "model.backbone.model.embedder.embedder.normalization.running_var"))
    # stages
    for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            # shortcut
            if layer_idx == 0:
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.0.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.1.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.1.bias",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.bias",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.1.running_mean",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.running_mean",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.1.running_var",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.running_var",
                    )
                )
            # 3 convs
            for i in range(3):
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.conv{i+1}.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn{i+1}.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn{i+1}.bias",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.bias",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn{i+1}.running_mean",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.running_mean",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn{i+1}.running_var",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.running_var",
                    )
                )
    # transformer encoder
    for i in range(config.encoder_layers):
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.sampling_offsets.weight", f"model.encoder.layers.{i}.self_attn.sampling_offsets.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.sampling_offsets.bias", f"model.encoder.layers.{i}.self_attn.sampling_offsets.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.attention_weights.weight", f"model.encoder.layers.{i}.self_attn.attention_weights.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.attention_weights.bias", f"model.encoder.layers.{i}.self_attn.attention_weights.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.value_proj.weight", f"model.encoder.layers.{i}.self_attn.value_proj.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.value_proj.bias", f"model.encoder.layers.{i}.self_attn.value_proj.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.output_proj.weight", f"model.encoder.layers.{i}.self_attn.output_proj.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.output_proj.bias", f"model.encoder.layers.{i}.self_attn.output_proj.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.norm1.weight", f"model.encoder.layers.{i}.self_attn_layer_norm.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.norm1.bias", f"model.encoder.layers.{i}.self_attn_layer_norm.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.linear1.weight", f"model.encoder.layers.{i}.fc1.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.linear1.bias", f"model.encoder.layers.{i}.fc1.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.linear2.weight", f"model.encoder.layers.{i}.fc2.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.linear2.bias", f"model.encoder.layers.{i}.fc2.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.norm2.weight", f"model.encoder.layers.{i}.final_layer_norm.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.norm2.bias", f"model.encoder.layers.{i}.final_layer_norm.bias"))

    # transformer decoder
    for i in range(config.decoder_layers):
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.sampling_offsets.weight", f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.sampling_offsets.bias", f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.attention_weights.weight", f"model.decoder.layers.{i}.encoder_attn.attention_weights.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.attention_weights.bias", f"model.decoder.layers.{i}.encoder_attn.attention_weights.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.value_proj.weight", f"model.decoder.layers.{i}.encoder_attn.value_proj.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.value_proj.bias", f"model.decoder.layers.{i}.encoder_attn.value_proj.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.output_proj.weight", f"model.decoder.layers.{i}.encoder_attn.output_proj.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.output_proj.bias", f"model.decoder.layers.{i}.encoder_attn.output_proj.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm1.weight", f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm1.bias", f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"model.decoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"model.decoder.layers.{i}.self_attn.out_proj.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm2.weight", f"model.decoder.layers.{i}.self_attn_layer_norm.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm2.bias", f"model.decoder.layers.{i}.self_attn_layer_norm.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.linear1.weight", f"model.decoder.layers.{i}.fc1.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.linear1.bias", f"model.decoder.layers.{i}.fc1.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.linear2.weight", f"model.decoder.layers.{i}.fc2.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.linear2.bias", f"model.decoder.layers.{i}.fc2.bias"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm3.weight", f"model.decoder.layers.{i}.final_layer_norm.weight"))
        rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"model.decoder.layers.{i}.final_layer_norm.bias"))

    # fmt: on

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def read_in_decoder_q_k_v(state_dict, config):
    # transformer decoder self-attention layers
    hidden_size = config.d_model
    for i in range(config.decoder_layers):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:hidden_size]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-hidden_size:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_deta_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DETA structure.
    """

    # load config
    config = get_deta_config()

    # load original state dict
    if model_name == "deta-resnet-50":
        filename = "adet_checkpoint0011.pth"
    elif model_name == "deta-resnet-50-24-epochs":
        filename = "adet_2x_checkpoint0023.pth"
    else:
        raise ValueError(f"Model name {model_name} not supported")
    checkpoint_path = hf_hub_download(repo_id="nielsr/deta-checkpoints", filename=filename)
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_decoder_q_k_v(state_dict, config)

    # fix some prefixes
    for key in state_dict.copy().keys():
        if "transformer.decoder.class_embed" in key or "transformer.decoder.bbox_embed" in key:
            val = state_dict.pop(key)
            state_dict[key.replace("transformer.decoder", "model.decoder")] = val
        if "input_proj" in key:
            val = state_dict.pop(key)
            state_dict["model." + key] = val
        if "level_embed" in key or "pos_trans" in key or "pix_trans" in key or "enc_output" in key:
            val = state_dict.pop(key)
            state_dict[key.replace("transformer", "model")] = val

    # finally, create HuggingFace model and load state dict
    model = DetaForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # load image processor
    processor = DetaImageProcessor(format="coco_detection")

    # verify our conversion on image
    img = prepare_img()
    encoding = processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values.to(device))

    # verify logits
    if model_name == "deta-resnet-50":
        expected_logits = torch.tensor(
            [[-7.3978, -2.5406, -4.1668], [-8.2684, -3.9933, -3.8096], [-7.0515, -3.7973, -5.8516]]
        )
        expected_boxes = torch.tensor([[0.5043, 0.4973, 0.9998], [0.2542, 0.5489, 0.4748], [0.5490, 0.2765, 0.0570]])
    elif model_name == "deta-resnet-50-24-epochs":
        expected_logits = torch.tensor(
            [[-7.1688, -2.4857, -4.8669], [-7.8630, -3.8154, -4.2674], [-7.2730, -4.1865, -5.5323]]
        )
        expected_boxes = torch.tensor([[0.5021, 0.4971, 0.9994], [0.2546, 0.5486, 0.4731], [0.1686, 0.1986, 0.2142]])

    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits.to(device), atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes.to(device), atol=1e-4)
    print("Everything ok!")

    if pytorch_dump_folder_path:
        # Save model and processor
        logger.info(f"Saving PyTorch model and processor to {pytorch_dump_folder_path}...")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    # Push to hub
    if push_to_hub:
        print("Pushing model and processor to hub...")
        model.push_to_hub(f"jozhang97/{model_name}")
        processor.push_to_hub(f"jozhang97/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="deta-resnet-50",
        choices=["deta-resnet-50", "deta-resnet-50-24-epochs"],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the folder to output PyTorch model.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()
    convert_deta_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
