# coding=utf-8
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


def convert_classifier_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    keys = list(state_dict.keys())
    for key in keys:
        if ".norm" in key:
            new_key = key.replace(".norm1.", ".layer_norm1.").replace(".norm2.", ".layer_norm2.")
            state_dict[new_key] = state_dict.pop(key)
        elif "qkv" in key and ".blocks." in key:
            tensor = state_dict.pop(key)
            q, k, v = torch.chunk(tensor, 3, dim=0)
            state_dict[key.replace(".attn.qkv.", ".self_attn.q_proj.")] = q
            state_dict[key.replace(".attn.qkv.", ".self_attn.k_proj.")] = k
            state_dict[key.replace(".attn.qkv.", ".self_attn.v_proj.")] = v
        elif ".attn.proj." in key:
            new_key = key.replace(".attn.proj.", ".self_attn.out_proj.")
            state_dict[new_key] = state_dict.pop(key)
        elif ".xattn.q." in key:
            new_key = key.replace(".xattn.q.", ".cross_attn.q_proj.")
            state_dict[new_key] = state_dict.pop(key)
        elif ".xattn.kv." in key:
            tensor = state_dict.pop(key)
            k, v = torch.chunk(tensor, 2, dim=0)
            state_dict[key.replace(".xattn.kv.", ".cross_attn.k_proj.")] = k
            state_dict[key.replace(".xattn.kv.", ".cross_attn.v_proj.")] = v

    state_dict = {k.replace(".blocks.", ".self_attention_layers."): v for k, v in state_dict.items()}
    state_dict = {k.replace(".cross_attention_block.", ".cross_attention_layer."): v for k, v in state_dict.items()}
    state_dict = {k.replace("linear.", "classifier."): v for k, v in state_dict.items()}
    return state_dict


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
    converted_state_dict = convert_classifier_state_dict(state_dict)

    result = model.load_state_dict(converted_state_dict, strict=False)
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
    # parser.add_argument("--model_name", type=str, default="vjepa2-vitg-fpc64-384-ssv2")
    # parser.add_argument("--model_name", type=str, default="vjepa2-vitl-fpc16-256-ssv2")
    # parser.add_argument("--model_name", type=str, default="vjepa2-vitl-fpc32-256-diving48")
    parser.add_argument("--model_name", type=str, default="vjepa2-vitg-fpc32-384-diving48")
    parser.add_argument("--base_dir", type=str, default="converted_models/")
    parser.add_argument("--repo_org", type=str, default="qubvel-hf")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--skip_verification", action="store_true")
    args = parser.parse_args()

    main(args)
