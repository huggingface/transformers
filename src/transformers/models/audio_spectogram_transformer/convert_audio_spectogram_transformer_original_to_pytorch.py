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
"""Convert Audio Spectogram Transformer checkpoints from the original repository. URL: https://github.com/YuanGongND/ast"""


import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import AudioSpectogramTransformerConfig, AudioSpectogramTransformerForSequenceClassification
from transformers.utils import logging

import torchaudio


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_audio_spectogram_transformer_config(model_name):
    config = AudioSpectogramTransformerConfig()

    config.num_labels = 527
    
    #TODO add id2label mappings
    # repo_id = "huggingface/label-files"
    # filename = "coco-detection-id2label.json"
    # id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # id2label = {int(k): v for k, v in id2label.items()}
    # config.id2label = id2label
    # config.label2id = {v: k for k, v in id2label.items()}

    return config


def rename_key(name):
    if "module.v" in name:
        name = name.replace("module.v", "audio_spectogram_transformer")
    if "cls_token" in name:
        name = name.replace("cls_token", "embeddings.cls_token")
    if "dist_token" in name:
        name = name.replace("dist_token", "embeddings.distillation_token")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "embeddings.position_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    # transformer blocks
    if "blocks" in name:
        name = name.replace("blocks", "encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    # final layernorm
    if "audio_spectogram_transformer.norm" in name:
        name = name.replace("audio_spectogram_transformer.norm", "audio_spectogram_transformer.layernorm")
    # classifier head
    if "module.mlp_head.0" in name:
        name = name.replace("module.mlp_head.0", "layernorm")
    if "module.mlp_head.1" in name:
        name = name.replace("module.mlp_head.1", "classifier")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[3])
            dim = config.hidden_size
            if "weight" in key:
                orig_state_dict[f"audio_spectogram_transformer.encoder.layer.{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                orig_state_dict[f"audio_spectogram_transformer.encoder.layer.{layer_num}.attention.attention.key.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[f"audio_spectogram_transformer.encoder.layer.{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"audio_spectogram_transformer.encoder.layer.{layer_num}.attention.attention.query.bias"] = val[:dim]
                orig_state_dict[f"audio_spectogram_transformer.encoder.layer.{layer_num}.attention.attention.key.bias"] = val[dim : dim * 2]
                orig_state_dict[f"audio_spectogram_transformer.encoder.layer.{layer_num}.attention.attention.value.bias"] = val[-dim:]
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def remove_keys(state_dict):
    ignore_keys = [
        "module.v.head.weight",
        "module.v.head.bias",
        "module.v.head_dist.weight",
        "module.v.head_dist.bias",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


@torch.no_grad()
def convert_audio_spectogram_transformer_checkpoint(model_name, checkpoint_url, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our YOLOS structure.
    """
    config = get_audio_spectogram_transformer_config(model_name)

    # load original state_dict
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # remove some keys
    remove_keys(state_dict)
    # rename some keys
    new_state_dict = convert_state_dict(state_dict, config)

    # load 🤗 model
    model = AudioSpectogramTransformerForSequenceClassification(config)
    model.eval()
    
    model.load_state_dict(new_state_dict)

    # verify outputs on dummy input
    filepath = hf_hub_download(repo_id="nielsr/audio-spectogram-transformer-checkpoint",
                           filename="sample_audio.flac",
                           repo_type="dataset")
    features = make_features(filepath, mel_bins=128) # shape(1024, 128)
    input_values = features.expand(1, 1024, 128)  # (batch_size, time, freq)  
    
    # forward pass
    outputs = model(input_values)
    logits = outputs.logits

    print("Shape of logits:", logits.shape)
    print("Predicted class:", logits.argmax(-1))

    expected_slice = torch.tensor([-0.8760, -7.0042, -8.6602])
    if not torch.allclose(logits[0, :3], expected_slice, atol=1e-4):
        raise ValueError("Logits don't match")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model_mapping = {
            "audio_spectogram_transformer_ti": "audio_spectogram_transformer-tiny",
            "audio_spectogram_transformer_s_200_pre": "audio_spectogram_transformer-small",
            "audio_spectogram_transformer_s_300_pre": "audio_spectogram_transformer-small-300",
            "audio_spectogram_transformer_s_dWr": "audio_spectogram_transformer-small-dwr",
            "audio_spectogram_transformer_base": "audio_spectogram_transformer-base",
        }

        print("Pushing to the hub...")
        model_name = model_mapping[model_name]
        model.push_to_hub(model_name, organization="hustvl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="audio_spectogram_transformer_s_200_pre",
        type=str,
        help=(
            "Name of the YOLOS model you'd like to convert. Should be one of 'audio_spectogram_transformer_ti', 'audio_spectogram_transformer_s_200_pre',"
            " 'audio_spectogram_transformer_s_300_pre', 'audio_spectogram_transformer_s_dWr', 'audio_spectogram_transformer_base'."
        ),
    )
    parser.add_argument(
        "--checkpoint_url", default="https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1", type=str, help="URL of the original state dict (.pth file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_audio_spectogram_transformer_checkpoint(args.model_name, args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)