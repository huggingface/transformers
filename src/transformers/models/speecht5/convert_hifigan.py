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
"""Convert SpeechT5 HiFi-GAN checkpoint."""

import argparse

import numpy as np
import torch

from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig, logging


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.speecht5")


def load_weights(checkpoint, hf_model, config):
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


@torch.no_grad()
def convert_hifigan_checkpoint(
    checkpoint_path,
    stats_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
):
    if config_path is not None:
        config = SpeechT5HifiGanConfig.from_pretrained(config_path)
    else:
        config = SpeechT5HifiGanConfig()

    model = SpeechT5HifiGan(config)

    orig_checkpoint = torch.load(checkpoint_path, weights_only=True)
    load_weights(orig_checkpoint["model"]["generator"], model, config)

    stats = np.load(stats_path)
    mean = stats[0].reshape(-1)
    scale = stats[1].reshape(-1)
    model.mean = torch.from_numpy(mean).float()
    model.scale = torch.from_numpy(scale).float()

    model.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--stats_path", required=True, default=None, type=str, help="Path to stats.npy file")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_hifigan_checkpoint(
        args.checkpoint_path,
        args.stats_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.push_to_hub,
    )
