# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Convert FastSpeech2Conformer HiFi-GAN checkpoint."""

import argparse
from pathlib import Path

import torch
import yaml

from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig, logging


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.FastSpeech2Conformer")


def load_weights(checkpoint, hf_model, config):
    vocoder_key_prefix = "tts.generator.vocoder."
    checkpoint = {k.replace(vocoder_key_prefix, ""): v for k, v in checkpoint.items() if vocoder_key_prefix in k}

    hf_model.apply_weight_norm()

    hf_model.conv_pre.weight_g.data = checkpoint["input_conv.weight_g"]
    hf_model.conv_pre.weight_v.data = checkpoint["input_conv.weight_v"]
    hf_model.conv_pre.bias.data = checkpoint["input_conv.bias"]

    for i in range(len(config.upsample_rates)):
        hf_model.upsampler[i].weight_g.data = checkpoint[f"upsamples.{i}.1.weight_g"]
        hf_model.upsampler[i].weight_v.data = checkpoint[f"upsamples.{i}.1.weight_v"]
        hf_model.upsampler[i].bias.data = checkpoint[f"upsamples.{i}.1.bias"]

    for i in range(len(config.upsample_rates) * len(config.resblock_kernel_sizes)):
        for j in range(len(config.resblock_dilation_sizes)):
            hf_model.resblocks[i].convs1[j].weight_g.data = checkpoint[f"blocks.{i}.convs1.{j}.1.weight_g"]
            hf_model.resblocks[i].convs1[j].weight_v.data = checkpoint[f"blocks.{i}.convs1.{j}.1.weight_v"]
            hf_model.resblocks[i].convs1[j].bias.data = checkpoint[f"blocks.{i}.convs1.{j}.1.bias"]

            hf_model.resblocks[i].convs2[j].weight_g.data = checkpoint[f"blocks.{i}.convs2.{j}.1.weight_g"]
            hf_model.resblocks[i].convs2[j].weight_v.data = checkpoint[f"blocks.{i}.convs2.{j}.1.weight_v"]
            hf_model.resblocks[i].convs2[j].bias.data = checkpoint[f"blocks.{i}.convs2.{j}.1.bias"]

    hf_model.conv_post.weight_g.data = checkpoint["output_conv.1.weight_g"]
    hf_model.conv_post.weight_v.data = checkpoint["output_conv.1.weight_v"]
    hf_model.conv_post.bias.data = checkpoint["output_conv.1.bias"]

    hf_model.remove_weight_norm()


def remap_hifigan_yaml_config(yaml_config_path):
    with Path(yaml_config_path).open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
        args = argparse.Namespace(**args)

    vocoder_type = args.tts_conf["vocoder_type"]
    if vocoder_type != "hifigan_generator":
        raise TypeError(f"Vocoder config must be for `hifigan_generator`, but got {vocoder_type}")

    remapped_dict = {}
    vocoder_params = args.tts_conf["vocoder_params"]

    # espnet_config_key -> hf_config_key
    key_mappings = {
        "channels": "upsample_initial_channel",
        "in_channels": "model_in_dim",
        "resblock_dilations": "resblock_dilation_sizes",
        "resblock_kernel_sizes": "resblock_kernel_sizes",
        "upsample_kernel_sizes": "upsample_kernel_sizes",
        "upsample_scales": "upsample_rates",
    }
    for espnet_config_key, hf_config_key in key_mappings.items():
        remapped_dict[hf_config_key] = vocoder_params[espnet_config_key]
    remapped_dict["sampling_rate"] = args.tts_conf["sampling_rate"]
    remapped_dict["normalize_before"] = False
    remapped_dict["leaky_relu_slope"] = vocoder_params["nonlinear_activation_params"]["negative_slope"]

    return remapped_dict


@torch.no_grad()
def convert_hifigan_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    yaml_config_path=None,
    repo_id=None,
):
    if yaml_config_path is not None:
        config_kwargs = remap_hifigan_yaml_config(yaml_config_path)
        config = FastSpeech2ConformerHifiGanConfig(**config_kwargs)
    else:
        config = FastSpeech2ConformerHifiGanConfig()

    model = FastSpeech2ConformerHifiGan(config)

    orig_checkpoint = torch.load(checkpoint_path, weights_only=True)
    load_weights(orig_checkpoint, model, config)

    model.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--yaml_config_path", default=None, type=str, help="Path to config.yaml of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_hifigan_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.yaml_config_path,
        args.push_to_hub,
    )
