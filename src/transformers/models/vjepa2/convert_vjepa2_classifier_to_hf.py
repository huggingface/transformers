# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import os
import re

import numpy as np
import torch
from decord import VideoReader
from huggingface_hub import HfApi, hf_hub_download

from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_video():
    path = hf_hub_download(
        repo_id="nateraw/kinetics-mini",
        filename="val/bowling/-WH-lxmGJVY_000005_000015.mp4",
        repo_type="dataset",
    )
    video_reader = VideoReader(path)
    return video_reader


CLASSIFIERS = {
    # Something-Something-v2 dataset
    "vjepa2-vitl-fpc16-256-ssv2": {
        "base_model": "facebook/vjepa2-vitl-fpc64-256",
        "checkpoint": "https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitl-16x2x3.pt",
        "num_labels": 174,
        "frames_per_clip": 16,
        "dataset": "something-something-v2",
        "result": (145, 0.30867, "Stuffing [something] into [something]"),
    },
    "vjepa2-vitg-fpc64-384-ssv2": {
        "base_model": "facebook/vjepa2-vitg-fpc64-384",
        "checkpoint": "https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt",
        "frames_per_clip": 64,
        "num_labels": 174,
        "dataset": "something-something-v2",
        "result": (112, 0.26408, "Putting [something] onto [something]"),
    },
    # Diving48 dataset
    "vjepa2-vitl-fpc32-256-diving48": {
        "base_model": "facebook/vjepa2-vitl-fpc64-256",
        "checkpoint": "https://dl.fbaipublicfiles.com/vjepa2/evals/diving48-vitl-256.pt",
        "num_labels": 48,
        "frames_per_clip": 32,
        "dataset": "diving48",
        "result": (35, 0.32875, "['Inward', '35som', 'NoTwis', 'TUCK']"),
    },
    "vjepa2-vitg-fpc32-384-diving48": {
        "base_model": "facebook/vjepa2-vitg-fpc64-384",
        "checkpoint": "https://dl.fbaipublicfiles.com/vjepa2/evals/diving48-vitg-384-32x4x3.pt",
        "frames_per_clip": 32,
        "num_labels": 48,
        "dataset": "diving48",
        "result": (22, 0.35351, "['Forward', '25som', '2Twis', 'PIKE']"),
    },
}

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"module.pooler.query_tokens":                          r"pooler.query_tokens",
    r"module.pooler.cross_attention_block.norm(\d+).":      r"pooler.cross_attention_layer.layer_norm\1.",
    r"module.pooler.cross_attention_block.xattn.(q|k|v).":  r"pooler.cross_attention_layer.cross_attn.\1_proj.",
    r"module.pooler.cross_attention_block.mlp.fc(\d+).":    r"pooler.cross_attention_layer.mlp.fc\1.",
    r"module.pooler.blocks.(\d+).norm(\d+).":               r"pooler.self_attention_layers.\1.layer_norm\2.",
    r"module.pooler.blocks.(\d+).attn.(q|k|v).":            r"pooler.self_attention_layers.\1.self_attn.\2_proj.",
    r"module.pooler.blocks.(\d+).attn.proj.":               r"pooler.self_attention_layers.\1.self_attn.out_proj.",
    r"module.pooler.blocks.(\d+).mlp.fc(\d+).":             r"pooler.self_attention_layers.\1.mlp.fc\2.",
    r"module.linear.":                                      r"classifier.",
}
# fmt: on


def get_id2label_mapping(dataset_name: str) -> dict[int, str]:
    path = hf_hub_download(
        repo_id="huggingface/label-files",
        filename=f"{dataset_name}-id2label.json",
        repo_type="dataset",
    )
    with open(path, "r") as f:
        id2label = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}
    return id2label


def split_qkv(state_dict):
    state_dict = state_dict.copy()
    keys = list(state_dict.keys())
    for key in keys:
        if ".qkv." in key:
            tensor = state_dict.pop(key)
            q, k, v = torch.chunk(tensor, 3, dim=0)
            state_dict[key.replace(".qkv.", ".q.")] = q
            state_dict[key.replace(".qkv.", ".k.")] = k
            state_dict[key.replace(".qkv.", ".v.")] = v
        elif ".kv." in key:
            tensor = state_dict.pop(key)
            k, v = torch.chunk(tensor, 2, dim=0)
            state_dict[key.replace(".kv.", ".k.")] = k
            state_dict[key.replace(".kv.", ".v.")] = v

    return state_dict


def convert_old_keys_to_new_keys(state_dict):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    old_text = "\n".join(state_dict)
    new_text = old_text
    for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        if replacement is None:
            new_text = re.sub(pattern, "", new_text)  # an empty line
            continue
        new_text = re.sub(pattern, replacement, new_text)
    output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def main(args: argparse.Namespace):
    model_params = CLASSIFIERS[args.model_name]
    id2label = get_id2label_mapping(model_params["dataset"])

    if not len(id2label) == model_params["num_labels"]:
        raise ValueError(
            f"Number of labels in id2label mapping ({len(id2label)}) does not "
            f"match number of labels in model ({model_params['num_labels']})"
        )

    model = VJEPA2ForVideoClassification.from_pretrained(
        model_params["base_model"],
        num_labels=model_params["num_labels"],
        id2label=id2label,
        frames_per_clip=model_params["frames_per_clip"],
    )
    processor = VJEPA2VideoProcessor.from_pretrained(model_params["base_model"])

    # load and convert classifier checkpoint
    checkpoint = torch.hub.load_state_dict_from_url(model_params["checkpoint"])
    state_dict = checkpoint["classifiers"][0]

    state_dict_qkv_split = split_qkv(state_dict)
    key_mapping = convert_old_keys_to_new_keys(state_dict_qkv_split.keys())
    converted_state_dict2 = {key_mapping[k]: v for k, v in state_dict_qkv_split.items()}

    result = model.load_state_dict(converted_state_dict2, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Error loading state dict: {result.unexpected_keys}")

    if not args.skip_verification:
        # get inputs
        video_reader = get_video()
        frame_indexes = np.arange(0, 128, 128 / model_params["frames_per_clip"])
        video = video_reader.get_batch(frame_indexes).asnumpy()
        inputs = processor(video, return_tensors="pt").to(device)

        # run model
        model.to(device).eval()
        with torch.no_grad():
            outputs = model(**inputs)

        # compare results
        probs = torch.softmax(outputs.logits, dim=-1)
        top_prob, top_idx = probs.topk(1)
        top_prob, top_idx = top_prob.item(), top_idx.item()
        label = id2label[top_idx]
        expected_id, expected_prob, expected_label = model_params["result"]

        if not top_idx == expected_id:
            raise ValueError(f"Expected id {expected_id} but got {top_idx}")
        if not label == expected_label:
            raise ValueError(f"Expected label {expected_label} but got {label}")
        if not np.isclose(top_prob, expected_prob, atol=1e-3):
            raise ValueError(f"Expected prob {expected_prob} but got {top_prob}")
        print("Verification passed")

    output_dir = os.path.join(args.base_dir, args.model_name)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    if args.push_to_hub:
        api = HfApi()
        repo_id = f"{args.repo_org}/{args.model_name}"
        if not api.repo_exists(repo_id):
            api.create_repo(repo_id, repo_type="model")
        api.upload_folder(folder_path=output_dir, repo_id=repo_id, repo_type="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="converted_models/")
    parser.add_argument("--repo_org", type=str, default="qubvel-hf")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--skip_verification", action="store_true")
    args = parser.parse_args()

    main(args)
