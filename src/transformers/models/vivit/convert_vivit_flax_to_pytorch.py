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
"""Convert Flax ViViT checkpoints from the original repository to PyTorch. URL:
https://github.com/google-research/scenic/tree/main/scenic/projects/vivit
"""

import argparse
import json
import os.path
from collections import OrderedDict

import numpy as np
import requests
import torch
from flax.training.checkpoints import restore_checkpoint
from huggingface_hub import hf_hub_download

from transformers import VivitConfig, VivitForVideoClassification, VivitImageProcessor
from transformers.image_utils import PILImageResampling


def download_checkpoint(path):
    url = "https://storage.googleapis.com/scenic-bucket/vivit/kinetics_400/vivit_base_16x2_unfactorized/checkpoint"

    with open(path, "wb") as f:
        with requests.get(url, stream=True) as req:
            for chunk in req.iter_content(chunk_size=2048):
                f.write(chunk)


def get_vivit_config() -> VivitConfig:
    config = VivitConfig()

    config.num_labels = 400
    repo_id = "huggingface/label-files"
    filename = "kinetics400-id2label.json"

    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    return config


# We will verify our results on a video of eating spaghetti
# Frame indices used: [ 47, 51, 55, 59, 63, 67, 71, 75, 80, 84, 88, 92, 96, 100, 104, 108, 113, 117,
# 121, 125, 129, 133, 137, 141, 146, 150, 154, 158, 162, 166, 170, 174]
def prepare_video():
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti_32_frames.npy", repo_type="dataset"
    )
    video = np.load(file)
    return list(video)


def transform_attention(current: np.ndarray):
    if np.ndim(current) == 2:
        return transform_attention_bias(current)

    elif np.ndim(current) == 3:
        return transform_attention_kernel(current)

    else:
        raise Exception(f"Invalid number of dimensions: {np.ndim(current)}")


def transform_attention_bias(current: np.ndarray):
    return current.flatten()


def transform_attention_kernel(current: np.ndarray):
    return np.reshape(current, (current.shape[0], current.shape[1] * current.shape[2])).T


def transform_attention_output_weight(current: np.ndarray):
    return np.reshape(current, (current.shape[0] * current.shape[1], current.shape[2])).T


def transform_state_encoder_block(state_dict, i):
    state = state_dict["optimizer"]["target"]["Transformer"][f"encoderblock_{i}"]

    prefix = f"encoder.layer.{i}."
    new_state = {
        prefix + "intermediate.dense.bias": state["MlpBlock_0"]["Dense_0"]["bias"],
        prefix + "intermediate.dense.weight": np.transpose(state["MlpBlock_0"]["Dense_0"]["kernel"]),
        prefix + "output.dense.bias": state["MlpBlock_0"]["Dense_1"]["bias"],
        prefix + "output.dense.weight": np.transpose(state["MlpBlock_0"]["Dense_1"]["kernel"]),
        prefix + "layernorm_before.bias": state["LayerNorm_0"]["bias"],
        prefix + "layernorm_before.weight": state["LayerNorm_0"]["scale"],
        prefix + "layernorm_after.bias": state["LayerNorm_1"]["bias"],
        prefix + "layernorm_after.weight": state["LayerNorm_1"]["scale"],
        prefix + "attention.attention.query.bias": transform_attention(
            state["MultiHeadDotProductAttention_0"]["query"]["bias"]
        ),
        prefix + "attention.attention.query.weight": transform_attention(
            state["MultiHeadDotProductAttention_0"]["query"]["kernel"]
        ),
        prefix + "attention.attention.key.bias": transform_attention(
            state["MultiHeadDotProductAttention_0"]["key"]["bias"]
        ),
        prefix + "attention.attention.key.weight": transform_attention(
            state["MultiHeadDotProductAttention_0"]["key"]["kernel"]
        ),
        prefix + "attention.attention.value.bias": transform_attention(
            state["MultiHeadDotProductAttention_0"]["value"]["bias"]
        ),
        prefix + "attention.attention.value.weight": transform_attention(
            state["MultiHeadDotProductAttention_0"]["value"]["kernel"]
        ),
        prefix + "attention.output.dense.bias": state["MultiHeadDotProductAttention_0"]["out"]["bias"],
        prefix + "attention.output.dense.weight": transform_attention_output_weight(
            state["MultiHeadDotProductAttention_0"]["out"]["kernel"]
        ),
    }

    return new_state


def get_n_layers(state_dict):
    return sum([1 if "encoderblock_" in k else 0 for k in state_dict["optimizer"]["target"]["Transformer"].keys()])


def transform_state(state_dict, classification_head=False):
    transformer_layers = get_n_layers(state_dict)

    new_state = OrderedDict()

    new_state["layernorm.bias"] = state_dict["optimizer"]["target"]["Transformer"]["encoder_norm"]["bias"]
    new_state["layernorm.weight"] = state_dict["optimizer"]["target"]["Transformer"]["encoder_norm"]["scale"]

    new_state["embeddings.patch_embeddings.projection.weight"] = np.transpose(
        state_dict["optimizer"]["target"]["embedding"]["kernel"], (4, 3, 0, 1, 2)
    )
    new_state["embeddings.patch_embeddings.projection.bias"] = state_dict["optimizer"]["target"]["embedding"]["bias"]

    new_state["embeddings.cls_token"] = state_dict["optimizer"]["target"]["cls"]
    new_state["embeddings.position_embeddings"] = state_dict["optimizer"]["target"]["Transformer"]["posembed_input"][
        "pos_embedding"
    ]

    for i in range(transformer_layers):
        new_state.update(transform_state_encoder_block(state_dict, i))

    if classification_head:
        new_state = {"vivit." + k: v for k, v in new_state.items()}
        new_state["classifier.weight"] = np.transpose(state_dict["optimizer"]["target"]["output_projection"]["kernel"])
        new_state["classifier.bias"] = np.transpose(state_dict["optimizer"]["target"]["output_projection"]["bias"])

    return {k: torch.tensor(v) for k, v in new_state.items()}


# checks that image processor settings are the same as in the original implementation
# original: https://github.com/google-research/scenic/blob/main/scenic/projects/vivit/data/video_tfrecord_dataset.py
# dataset specific config:
# https://github.com/google-research/scenic/blob/main/scenic/projects/vivit/configs/kinetics400/vivit_base_k400.py
def get_processor() -> VivitImageProcessor:
    extractor = VivitImageProcessor()

    assert extractor.do_resize is True
    assert extractor.size == {"shortest_edge": 256}
    assert extractor.do_center_crop is True
    assert extractor.crop_size == {"width": 224, "height": 224}
    assert extractor.resample == PILImageResampling.BILINEAR

    # here: https://github.com/deepmind/dmvr/blob/master/dmvr/modalities.py
    # one can seen that add_image has default values for normalization_mean and normalization_std set to 0 and 1
    # which effectively means no normalization (and ViViT does not overwrite those when calling this func)
    assert extractor.do_normalize is False
    assert extractor.do_rescale is True
    assert extractor.rescale_factor == 1 / 255

    # zero-centering = True in original implementation
    assert extractor.do_zero_centering is True

    return extractor


def convert(output_path: str):
    flax_model_path = "checkpoint"

    if not os.path.exists(flax_model_path):
        download_checkpoint(flax_model_path)

    state_dict = restore_checkpoint(flax_model_path, None)
    new_state = transform_state(state_dict, classification_head=True)

    config = get_vivit_config()

    assert config.image_size == 224
    assert config.num_frames == 32

    model = VivitForVideoClassification(config)
    model.load_state_dict(new_state)
    model.eval()

    extractor = get_processor()

    video = prepare_video()
    inputs = extractor(video, return_tensors="pt")

    outputs = model(**inputs)

    expected_shape = torch.Size([1, 400])
    expected_slice = torch.tensor([-1.0543, 2.0764, -0.2104, 0.4439, -0.9658])

    assert outputs.logits.shape == expected_shape
    assert torch.allclose(outputs.logits[0, :5], expected_slice, atol=1e-4), outputs.logits[0, :5]

    model.save_pretrained(output_path)
    extractor.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_model_name", "-o", type=str, help="Output path for the converted HuggingFace model")

    args = parser.parse_args()
    convert(args.output_model_name)
