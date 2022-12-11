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
"""Convert TimeSformer checkpoints from the original repository: https://github.com/MCG-NJU/TimeSformer"""

import argparse
import json

import numpy as np
import torch

import gdown
from huggingface_hub import hf_hub_download
from transformers import TimesformerConfig, TimesformerForVideoClassification, VideoMAEFeatureExtractor


def get_timesformer_config(model_name):
    config = TimesformerConfig()

    if "large" in model_name:
        config.num_frames = 96

    if "hr" in model_name:
        config.num_frames = 16
        config.image_size = 448

    repo_id = "huggingface/label-files"
    if "k400" in model_name:
        config.num_labels = 400
        filename = "kinetics400-id2label.json"
    elif "k600" in model_name:
        config.num_labels = 600
        filename = "kinetics600-id2label.json"
    elif "ssv2" in model_name:
        config.num_labels = 174
        filename = "something-something-v2-id2label.json"
    else:
        raise ValueError("Model name should either contain 'k400', 'k600' or 'ssv2'.")
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


def rename_key(name):
    if "encoder." in name:
        name = name.replace("encoder.", "")
    if "cls_token" in name:
        name = name.replace("cls_token", "timesformer.embeddings.cls_token")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "timesformer.embeddings.position_embeddings")
    if "time_embed" in name:
        name = name.replace("time_embed", "timesformer.embeddings.time_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "timesformer.embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "timesformer.embeddings.norm")
    if "blocks" in name:
        name = name.replace("blocks", "timesformer.encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name and "bias" not in name and "temporal" not in name:
        name = name.replace("attn", "attention.self")
    if "attn" in name and "temporal" not in name:
        name = name.replace("attn", "attention.attention")
    if "temporal_norm1" in name:
        name = name.replace("temporal_norm1", "temporal_layernorm")
    if "temporal_attn.proj" in name:
        name = name.replace("temporal_attn", "temporal_attention.output.dense")
    if "temporal_fc" in name:
        name = name.replace("temporal_fc", "temporal_dense")
    if "norm1" in name and "temporal" not in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "norm.weight" in name and "fc" not in name and "temporal" not in name:
        name = name.replace("norm.weight", "timesformer.layernorm.weight")
    if "norm.bias" in name and "fc" not in name and "temporal" not in name:
        name = name.replace("norm.bias", "timesformer.layernorm.bias")
    if "head" in name:
        name = name.replace("head", "classifier")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if key.startswith("model."):
            key = key.replace("model.", "")

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[1])
            prefix = "timesformer.encoder.layer."
            if "temporal" in key:
                postfix = ".temporal_attention.attention.qkv."
            else:
                postfix = ".attention.attention.qkv."
            if "weight" in key:
                orig_state_dict[f"{prefix}{layer_num}{postfix}weight"] = val
            else:
                orig_state_dict[f"{prefix}{layer_num}{postfix}bias"] = val
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


# We will verify our results on a video of eating spaghetti
# Frame indices used: [164 168 172 176 181 185 189 193 198 202 206 210 215 219 223 227]
def prepare_video():
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti.npy", repo_type="dataset"
    )
    video = np.load(file)
    return list(video)


def convert_timesformer_checkpoint(checkpoint_url, pytorch_dump_folder_path, model_name, push_to_hub):
    config = get_timesformer_config(model_name)

    model = TimesformerForVideoClassification(config)

    # download original checkpoint, hosted on Google Drive
    output = "pytorch_model.bin"
    gdown.cached_download(checkpoint_url, output, quiet=False)
    files = torch.load(output, map_location="cpu")
    if "model" in files:
        state_dict = files["model"]
    elif "module" in files:
        state_dict = files["module"]
    else:
        state_dict = files["model_state"]
    new_state_dict = convert_state_dict(state_dict, config)

    model.load_state_dict(new_state_dict)
    model.eval()

    # verify model on basic input
    feature_extractor = VideoMAEFeatureExtractor(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    video = prepare_video()
    inputs = feature_extractor(video[:8], return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits

    model_names = [
        # Kinetics-400 checkpoints (hr = high resolution input of 448px instead of 224px)
        "timesformer-base-finetuned-k400",
        "timesformer-large-finetuned-k400",
        "timesformer-hr-finetuned-k400",
        # Kinetics-600 checkpoints (hr = high resolution input of 448px instead of 224px)
        "timesformer-base-finetuned-k600",
        "timesformer-large-finetuned-k600",
        "timesformer-hr-finetuned-k600",
        # Something-Something-v2 checkpoints (hr = high resolution input of 448px instead of 224px)
        "timesformer-base-finetuned-ssv2",
        "timesformer-large-finetuned-ssv2",
        "timesformer-hr-finetuned-ssv2",
    ]

    # NOTE: logits were tested with image_mean and image_std equal to [0.5, 0.5, 0.5] and [0.5, 0.5, 0.5]
    if model_name == "timesformer-base-finetuned-k400":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([-0.3016, -0.7713, -0.4205])
    elif model_name == "timesformer-base-finetuned-k600":
        expected_shape = torch.Size([1, 600])
        expected_slice = torch.tensor([-0.7267, -0.7466, 3.2404])
    elif model_name == "timesformer-base-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([-0.9059, 0.6433, -3.1457])
    elif model_name == "timesformer-large-finetuned-k400":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0, 0, 0])
    elif model_name == "timesformer-large-finetuned-k600":
        expected_shape = torch.Size([1, 600])
        expected_slice = torch.tensor([0, 0, 0])
    elif model_name == "timesformer-large-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([0, 0, 0])
    elif model_name == "timesformer-hr-finetuned-k400":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([-0.9617, -3.7311, -3.7708])
    elif model_name == "timesformer-hr-finetuned-k600":
        expected_shape = torch.Size([1, 600])
        expected_slice = torch.tensor([2.5273, 0.7127, 1.8848])
    elif model_name == "timesformer-hr-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([-3.6756, -0.7513, 0.7180])
    else:
        raise ValueError(f"Model name not supported. Should be one of {model_names}")

    # verify logits
    assert logits.shape == expected_shape
    assert torch.allclose(logits[0, :3], expected_slice, atol=1e-4)
    print("Logits ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and feature extractor to {pytorch_dump_folder_path}")
        feature_extractor.save_pretrained(pytorch_dump_folder_path)
        model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        model.push_to_hub(f"fcakyon/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://drive.google.com/u/1/uc?id=17yvuYp9L4mn-HpIcK5Zo6K3UoOy1kA5l&export=download",
        type=str,
        help=(
            "URL of the original PyTorch checkpoint (on Google Drive) you'd like to convert. Should be a direct"
            " download link."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--model_name", default="timesformer-base-finetuned-k400", type=str, help="Name of the model.")
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_timesformer_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.model_name, args.push_to_hub
    )
