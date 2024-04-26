# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Convert Audio MAE checkpoints from the original repository: https://github.com/facebookresearch/mae"""

import argparse
from pathlib import Path
import pickle
import torch
import numpy as np
import random

from transformers import AudioMAEConfig, AudioMAEForPreTraining, AudioMAEFeatureExtractor
from datasets import load_dataset, Audio


seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
torch.use_deterministic_algorithms(True)

def rename_key(name):
    if "cls_token" in name:
        name = name.replace("cls_token", "audiomae.embeddings.cls_token")
    if "pos_embed" in name and "decoder" not in name:
        name = name.replace("pos_embed", "audiomae.embeddings.position_embeddings")
    if "mask_token" in name:
        name = name.replace("mask_token", "decoder.mask_token")
    if "decoder_pos_embed" in name:
        name = name.replace("decoder_pos_embed", "decoder.decoder_pos_embed")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "audiomae.embeddings.patch_embeddings.projection")
    if "decoder_blocks" in name:
        name = name.replace("decoder_blocks", "decoder.decoder_layers")
    if "blocks" in name:
        name = name.replace("blocks", "audiomae.encoder.layers")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name and 'meta' not in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name and 'meta' not in name:
        name = name.replace("mlp.fc2", "output.dense")
    #TODO need to convert meta_mlp into somebetter name.
    if 'meta_mlp.fc1' in name:
        name = name.replace('meta_mlp.fc1', 'meta_mlp.fc1')
    if 'meta_mlp.fc2' in name:
        name = name.replace('meta_mlp.fc2', 'meta_mlp.fc2')
    if "decoder_embed" in name:
        name = name.replace("decoder_embed", "decoder.decoder_embed")
    if "decoder_norm" in name:
        name = name.replace("decoder_norm", "decoder.decoder_norm")
    if "decoder_pred" in name:
        name = name.replace("decoder_pred", "decoder.decoder_pred")
    if "norm.weight" in name and "decoder" not in name:
        name = name.replace("norm.weight", "audiomae.layernorm.weight")
    if "norm.bias" in name and "decoder" not in name:
        name = name.replace("norm.bias", "audiomae.layernorm.bias")
    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[1])
            if "decoder_blocks" in key:
                dim = config.decoder_hidden_size
                prefix = "decoder.decoder_layers."
                if "weight" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.self.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.self.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.self.value.weight"] = val[-dim:, :]
                elif "bias" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.self.query.bias"] = val[:dim]
                    orig_state_dict[f"{prefix}{layer_num}.attention.self.key.bias"] = val[dim : dim * 2]
                    orig_state_dict[f"{prefix}{layer_num}.attention.self.value.bias"] = val[-dim:]
            else:
                dim = config.hidden_size
                prefix = "audiomae.encoder.layers."
                if "weight" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.self.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.self.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.self.value.weight"] = val[-dim:, :]
                elif "bias" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.self.query.bias"] = val[:dim]
                    orig_state_dict[f"{prefix}{layer_num}.attention.self.key.bias"] = val[dim : dim * 2]
                    orig_state_dict[f"{prefix}{layer_num}.attention.self.value.bias"] = val[-dim:]

        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict

def convert_audio_mae_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub=False):
    config = AudioMAEConfig()

    model = AudioMAEForPreTraining(config)

    # state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    checkpoint = torch.load('/Users/subhashp/Documents/Open-Source/AudioMAE/ckpt/pretrained.pth', map_location='cpu')
    state_dict = checkpoint['model']

    new_state_dict = convert_state_dict(state_dict, config)

    model.load_state_dict(new_state_dict)
    model.eval()
    
    #TODO we need to load an audio file, and compute the model logits from it.
    feature_extractor = AudioMAEFeatureExtractor()

    # Load the waveform from file
    with open('waveform.pickle', 'rb') as f:
        waveform = pickle.load(f)
    inputs = feature_extractor([waveform], sampling_rate=16000, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(dim=0)
    # print(inputs['pixel_values'].shape)
    # forward pass
    
    outputs = model(**inputs)
    logits = outputs.logits
    # print(logits)

    # #TODO update this expected slice value
    # expected_slice = torch.tensor(
    #     [[-0.9192, -0.8481, -1.1259], [-1.1349, -1.0034, -1.2599], [-1.1757, -1.0429, -1.2726]]
    # )

    # # verify logits
    # if not torch.allclose(logits[0, :3], expected_slice, atol=1e-4):
    #     raise ValueError("Logits don't match")
    # print("Looks ok!")
        
    # if pytorch_dump_folder_path is not None:
    #     Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    #     print(f"Saving model AudioMAE to {pytorch_dump_folder_path}")
    #     model.save_pretrained(pytorch_dump_folder_path)
    #     print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    #     feature_extractor.save_pretrained(pytorch_dump_folder_path)

    # if push_to_hub:
    #     print("Pushing model and feature extractor to the hub...")
    #     model.push_to_hub(f"MIT/AudioMAE")
    #     feature_extractor.push_to_hub(f"MIT/AudioMAE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu/view",
        type=str,
        help="URL of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    with torch.no_grad():
        convert_audio_mae_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
