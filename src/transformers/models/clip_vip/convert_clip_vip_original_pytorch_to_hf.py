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

import argparse

# from clip_vip import load
import torch

from transformers import CLIPViPConfig, CLIPViPModel


@torch.no_grad()
def convert_clip_vip_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = CLIPViPConfig.from_pretrained(config_path)
    else:
        config = CLIPViPConfig(projection_dim=512, text_config={}, vision_config={})

    hf_model = CLIPViPModel(config).eval()
    pt_model = torch.load(checkpoint_path)

    pt_model = {k.replace("clipmodel.", ""): v for k, v in pt_model.items()}
    print(pt_model["text_model.embeddings.position_ids"])
    print(pt_model["vision_model.embeddings.position_ids"])
    # we remove position_ids since they are no longer persistent / present in the
    # state_dicts of CLIP models
    del pt_model["text_model.embeddings.position_ids"]
    del pt_model["vision_model.embeddings.position_ids"]
    hf_model.eval()
    hf_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to original serialized state_dict")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()

    convert_clip_vip_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
