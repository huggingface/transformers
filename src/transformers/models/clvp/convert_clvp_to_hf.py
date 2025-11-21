# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

"""
Weights conversion script for CLVP
"""

import argparse
import os

import torch
from huggingface_hub import hf_hub_download

from transformers import ClvpConfig, ClvpModelForConditionalGeneration


_MODELS = {
    "clvp": "https://huggingface.co/jbetker/tortoise-tts-v2/blob/main/.models/clvp2.pth",
    "decoder": "https://huggingface.co/jbetker/tortoise-tts-v2/blob/main/.models/autoregressive.pth",
}

dim = 1024
sub_dim = dim // 16

CLVP_ENCODERS_MAPPING = {
    "text_transformer.transformer.attn_layers": "text_encoder_model",
    "speech_transformer.transformer.attn_layers": "speech_encoder_model",
    "text_transformer.transformer.norm": "text_encoder_model.final_layer_norm",
    "speech_transformer.transformer.norm": "speech_encoder_model.final_layer_norm",
    "to_text_latent": "text_encoder_model.projection",
    "to_speech_latent": "speech_encoder_model.projection",
    "text_emb": "text_encoder_model.token_embedding",
    "speech_emb": "speech_encoder_model.token_embedding",
    "1.wrap.net.0": "mlp.fc1",
    "1.wrap.net.3": "mlp.fc2",
    "1.wrap": "self_attn",
    "to_out": "out_proj",
    "to_q": "q_proj",
    "to_k": "k_proj",
    "to_v": "v_proj",
    "temperature": "logit_scale",
}

CLVP_DECODER_MAPPING = {
    "conditioning_encoder.init": "conditioning_encoder.mel_conv",
    "conditioning_encoder.attn": "conditioning_encoder.mel_attn_blocks",
    "mel_attn_blocks": "group_norms",
    ".norm.weight": ".weight",
    ".norm.bias": ".bias",
    "text_embedding": "conditioning_encoder.text_token_embedding",
    "text_pos_embedding.emb": "conditioning_encoder.text_position_embedding",
    "final_norm": "speech_decoder_model.final_norm",
    "mel_head": "speech_decoder_model.lm_head",
    "gpt.ln_f": "speech_decoder_model.model.decoder.layer_norm",
    "mel_embedding": "speech_decoder_model.model.decoder.input_embeds_layer",
    "mel_pos_embedding.emb": "speech_decoder_model.model.decoder.position_embeds_layer",
    "gpt.h": "speech_decoder_model.model.decoder.layers",
    "ln_1": "input_layernorm",
    "ln_2": "post_attention_layernorm",
}


def update_index(present_index):
    if present_index % 2 == 0:
        return int(present_index / 2)
    else:
        return int((present_index - 1) / 2)


def convert_encoder_weights(original_weights):
    converted_weights = {}
    original_weights_keys = sorted(original_weights.keys())
    for original_key in original_weights_keys:
        updated_key = original_key
        # for input_rmsnorm.weight and post_attention_rmsnorm.weight
        if "0.0.g" in updated_key:
            present_index = updated_key.split(".")[4]
            if int(present_index) % 2 == 0:
                updated_key = updated_key.replace("0.0.g", "input_rmsnorm.weight")
            else:
                updated_key = updated_key.replace("0.0.g", "post_attention_rmsnorm.weight")

        if "transformer.attn_layers.layers" in updated_key:
            present_index = updated_key.split(".")[4]
            updated_index = update_index(int(present_index))
            updated_key = updated_key.replace(
                f"transformer.attn_layers.layers.{present_index}", f"transformer.attn_layers.layers.{updated_index}"
            )

        for k, v in CLVP_ENCODERS_MAPPING.items():
            if k in updated_key:
                updated_key = updated_key.replace(k, v)

        converted_weights[updated_key] = original_weights.pop(original_key)

    return converted_weights


def convert_decoder_weights(original_weights):
    converted_weights = {}
    original_weights_keys = sorted(original_weights.keys())
    for original_key in original_weights_keys:
        updated_key = original_key
        if len(updated_key.split(".")) > 3:
            index, attr = updated_key.split(".")[2], updated_key.split(".")[-1]

        # for decoder attention
        if "attn.c_attn" in updated_key:
            if attr == "weight":
                slice1, slice2, slice3 = original_weights[updated_key].squeeze(-1).T.split(split_size=dim, dim=0)
            else:
                slice1, slice2, slice3 = original_weights[updated_key].split(split_size=dim, dim=0)
            converted_weights[f"speech_decoder_model.model.decoder.layers.{index}.attn.q_proj.{attr}"] = slice1
            converted_weights[f"speech_decoder_model.model.decoder.layers.{index}.attn.k_proj.{attr}"] = slice2
            converted_weights[f"speech_decoder_model.model.decoder.layers.{index}.attn.v_proj.{attr}"] = slice3
            continue

        if "attn.c_proj" in updated_key:
            converted_weights[f"speech_decoder_model.model.decoder.layers.{index}.attn.out_proj.{attr}"] = (
                original_weights[updated_key].squeeze(-1).T
            )
            continue

        if "attn.bias" in updated_key or "attn.masked_bias" in updated_key or "text_head" in updated_key:
            original_weights.pop(updated_key)
            continue

        # conditional encoder attention
        if "qkv" in updated_key:
            if attr == "weight":
                slice1, slice2, slice3 = original_weights[updated_key].squeeze(-1).split(split_size=dim, dim=0)
            else:
                slice1, slice2, slice3 = original_weights[updated_key].split(split_size=dim, dim=0)

            indices = torch.arange(dim)
            index1, index2, index3 = (
                indices.unfold(0, sub_dim, sub_dim * 3).flatten(),
                indices[sub_dim:].unfold(0, sub_dim, sub_dim * 3).flatten(),
                indices[2 * sub_dim :].unfold(0, sub_dim, sub_dim * 3).flatten(),
            )

            converted_weights[f"conditioning_encoder.mel_attn_blocks.{index}.q_proj.{attr}"] = torch.concatenate(
                [slice1[index1], slice2[index3], slice3[index2]],
                axis=0,
            )
            converted_weights[f"conditioning_encoder.mel_attn_blocks.{index}.k_proj.{attr}"] = torch.concatenate(
                [slice1[index2], slice2[index1], slice3[index3]],
                axis=0,
            )
            converted_weights[f"conditioning_encoder.mel_attn_blocks.{index}.v_proj.{attr}"] = torch.concatenate(
                [slice1[index3], slice2[index2], slice3[index1]],
                axis=0,
            )
            continue

        if "proj_out" in updated_key:
            converted_weights[f"conditioning_encoder.mel_attn_blocks.{index}.out_proj.{attr}"] = original_weights[
                updated_key
            ].squeeze(-1)
            continue

        for k, v in CLVP_DECODER_MAPPING.items():
            if k in updated_key:
                updated_key = updated_key.replace(k, v)

        converted_weights[updated_key] = original_weights.pop(original_key)

    return converted_weights


def _download(url: str, root: str):
    repo_id = f"{url.split('/')[3]}/{url.split('/')[4]}"
    filename = f"{url.split('/')[-2]}/{url.split('/')[-1]}"
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_filename=root,
        local_dir_use_symlinks=False,
    )


def convert_clvp_weights(checkpoint_path, pytorch_dump_folder_path):
    converted_checkpoint = {}

    for each_model_name, each_model_url in _MODELS.items():
        each_model_path = os.path.join(checkpoint_path, each_model_url.split("/")[-1])
        if not os.path.exists(each_model_path):
            print(f"\n{each_model_name} was not found! Downloading it to {each_model_path}")
            _download(url=each_model_url, root=each_model_path)

        if each_model_name == "clvp":
            clvp_checkpoint = torch.load(each_model_path, map_location="cpu")
        else:
            decoder_checkpoint = torch.load(each_model_path, map_location="cpu")

    # Converting the weights
    converted_checkpoint.update(**convert_encoder_weights(clvp_checkpoint))
    converted_checkpoint.update(**convert_decoder_weights(decoder_checkpoint))

    config = ClvpConfig.from_pretrained("susnato/clvp_dev")
    model = ClvpModelForConditionalGeneration(config)

    model.load_state_dict(converted_checkpoint, strict=True)
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Model saved at {pytorch_dump_folder_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # Required parameters
    parser.add_argument(
        "--checkpoint_path", type=str, help="Path to the folder of downloaded checkpoints. (Please enter full path)"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model. (Please enter full path)",
    )
    args = parser.parse_args()

    convert_clvp_weights(args.checkpoint_path, args.pytorch_dump_folder_path)
