# coding=utf-8
# Copyright 2022 Meta Platforms authors and The HuggingFace Team. All rights reserved.
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
import os

import torch

from transformers import FlavaConfig, FlavaForPreTraining
from transformers.models.flava.convert_dalle_to_flava_codebook import convert_dalle_checkpoint


def count_parameters(state_dict):
    # encoder.embeddings are double copied in original FLAVA
    return sum(param.float().sum() if "encoder.embeddings" not in key else 0 for key, param in state_dict.items())


def upgrade_state_dict(state_dict, codebook_state_dict):
    upgrade = {}

    for key, value in state_dict.items():
        if "text_encoder.embeddings" in key or "image_encoder.embeddings" in key:
            continue

        key = key.replace("heads.cmd.mim_head.cls.predictions", "mmm_image_head")
        key = key.replace("heads.cmd.mlm_head.cls.predictions", "mmm_text_head")
        key = key.replace("heads.cmd.itm_head.cls", "itm_head")
        key = key.replace("heads.cmd.itm_head.pooler", "itm_head.pooler")
        key = key.replace("heads.cmd.clip_head.logit_scale", "flava.logit_scale")
        key = key.replace("heads.fairseq_mlm.cls.predictions", "mlm_head")
        key = key.replace("heads.imagenet.mim_head.cls.predictions", "mim_head")
        key = key.replace("mm_text_projection", "flava.text_to_mm_projection")
        key = key.replace("mm_image_projection", "flava.image_to_mm_projection")
        key = key.replace("image_encoder.module", "flava.image_model")
        key = key.replace("text_encoder.module", "flava.text_model")
        key = key.replace("mm_encoder.module.encoder.cls_token", "flava.multimodal_model.cls_token")
        key = key.replace("mm_encoder.module", "flava.multimodal_model")
        key = key.replace("text_projection", "flava.text_projection")
        key = key.replace("image_projection", "flava.image_projection")

        upgrade[key] = value.float()

    for key, value in codebook_state_dict.items():
        upgrade[f"image_codebook.{key}"] = value

    return upgrade


@torch.no_grad()
def convert_flava_checkpoint(checkpoint_path, codebook_path, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = FlavaConfig.from_pretrained(config_path)
    else:
        config = FlavaConfig()

    hf_model = FlavaForPreTraining(config).eval()

    codebook_state_dict = convert_dalle_checkpoint(codebook_path, None, save_checkpoint=False)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    else:
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_path, map_location="cpu")

    hf_state_dict = upgrade_state_dict(state_dict, codebook_state_dict)
    hf_model.load_state_dict(hf_state_dict)
    hf_state_dict = hf_model.state_dict()
    hf_count = count_parameters(hf_state_dict)
    state_dict_count = count_parameters(state_dict) + count_parameters(codebook_state_dict)

    assert torch.allclose(hf_count, state_dict_count, atol=1e-3)

    hf_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to flava checkpoint")
    parser.add_argument("--codebook_path", default=None, type=str, help="Path to flava codebook checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()

    convert_flava_checkpoint(args.checkpoint_path, args.codebook_path, args.pytorch_dump_folder_path, args.config_path)
